from __future__ import division
from keras import Model
from keras.layers import *
from keras.models import load_model
from extras.layers import TemporalMaxPooling


model_seed = 1000
epsilon = 1e-6

# dummy variables
word2ids = {'<pad>': 0, 'hi': 1}  # TODO
char2ids = {'<pad>': 0, 'a': 1}  # TODO


# Notes:
# Can add Batch Normalizations
# Can add Bias regularizers
# Initialzers can be configurable
# Can add configurable RNN cells - currently no support for layers or diffrent cell type
# Configurable Similarity model losses like -> Pairwise Cosine, Hasdell, Triplet, etc
# Add learning rate scheduling, etc


class KerasModel(object):
    def __init__(self):
        self.model = None

    def compile_model(self, optimizer):
        raise NotImplementedError("Can't directly compile Base class, Inherit it"
                                  " and define a model using the encoder")

    def load(self, path):
        self.model = load_model(filepath=path)

    def save(self, path):
        self.model.save(path)


class DeepEncoder(object):
    def __init__(self, vocab_size, charset_size,
                 max_sequence_length, max_word_length,
                 use_words=True, char_kernel_sizes=(2, 3), word_embedding_size=64,
                 char_embedding_size=32, dropout=0.2, recurrent_dropout=0.2, encoder_hidden_units=64,
                 l2_regularization=0.01, bidirectional=True):
        self.use_words = use_words
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.char_kernel_sizes = char_kernel_sizes
        self.max_sequence_length = max_sequence_length
        self.max_word_length = max_word_length
        self.encoder_hidden_units = encoder_hidden_units

        self.vocab_size = vocab_size
        self.charset_size = charset_size

        # --- Layers ---
        self.word_embed = None
        if self.use_words:
            self.word_embed = Embedding(
                input_dim=self.vocab_size,
                input_length=self.max_sequence_length,
                output_dim=self.word_embedding_size,
                mask_zero=True,
                name='word_embed'
            )

        self.char_embed = Embedding(
            input_dim=self.charset_size,
            input_length=self.max_word_length,
            output_dim=self.char_embedding_size,
            name='char_embed'
            # mask_zero=True # Conv1D doesn't suppport masking
        )

        self._convs = [Conv1D(
            filters=self.word_embedding_size,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_regularization)
        ) for kernel_size in self.char_kernel_sizes]

        self.permute_as_021 = Permute(
            dims=(2, 1)
        )

        self.global_max_pool = GlobalMaxPool1D()
        self.masked_global_max_pool = TemporalMaxPooling()

        self.char_convolutions = [Lambda(
            function=self.char2word,
            arguments={'i': i},
            name='char_embed_conv_' + str(self.char_kernel_sizes[i])
        ) for i in range(len(self.char_kernel_sizes))]

        char2word_size = self.word_embedding_size * len(self.char_kernel_sizes)
        self.char2word_dense = TimeDistributed(
            Dense(
                units=char2word_size,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_regularization)
            ),
            input_shape=(max_sequence_length, char2word_size),
            name='dense_over_convolutions'
        )

        self.concat_char_convolutions = Concatenate(
            axis=2,
            name='concat_convolutions'
        )

        self.residual_add = Add(
            name='skip_connect_around_dense'
        )

        self.concat_embeddings = Concatenate(
            axis=2,
            name='concat_char2word_word_embedding'
        )

        self.encoder = LSTM(
            units=self.encoder_hidden_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name='recurrent_encoder',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_regularization)
        )

        if bidirectional:
            self.encoder = Bidirectional(
                self.encoder
            )

        self.multiply = Multiply()
        self.char_input_dropout = Dropout(rate=dropout, seed=model_seed)
        self.word_input_dropout = Dropout(rate=dropout, seed=model_seed)
        self.dense_dropout = Dropout(rate=dropout, seed=model_seed)

    def char2word(self, _char_inputs, i):
        # batch_size x max_sequence_length x max_word_length ->
        # (batch_size * max_sequence_length) x max_word_length
        char_inputs_reshaped = K.reshape(_char_inputs, shape=(-1, self.max_word_length))
        # (batch_size * max_sequence_length) x adjusted_max_word_length ->
        # (batch_size * max_sequence_length) x adjusted_max_word_length x char_embedding_size
        char_inputs_embedded = self.char_embed(char_inputs_reshaped)

        # We will put zero vector for 0 indices
        mask = K.cast(K.not_equal(char_inputs_reshaped, 0), 'float32')
        expanded_mask = K.expand_dims(mask, axis=2)
        tiled_mask = K.tile(expanded_mask, n=[1, 1, self.char_embedding_size])
        char_inputs_embedded_masked = self.multiply([tiled_mask,  char_inputs_embedded])
        # (batch_size * max_sequence_length) x adjusted_max_word_length x char_embedding_size ->
        # (batch_size * max_sequence_length) x word_embedding_size
        merged = self.global_max_pool(
            self._convs[i](
                self.char_input_dropout(
                    char_inputs_embedded_masked
                )
            )
        )
        # (batch_size * max_sequence_length) x word_embedding_size ->
        # batch_size x max_sequence_length x word_embedding_size
        return K.reshape(merged, shape=(-1, self.max_sequence_length, self.word_embedding_size))

    def encode(self, inp, char_inp):
        convolutions = []
        for convolution in self.char_convolutions:
            convolutions.append(convolution(char_inp))

        # [batch_size x max_sequence_length x word_embedding_size] ->
        # batch_size x max_sequence_length x (word_embedding_size * num_kernels)
        if len(self.char_kernel_sizes) > 1:
            char2word_embedded = self.concat_char_convolutions(convolutions)
        else:
            char2word_embedded = convolutions[0]

        # batch_size x max_sequence_length x (word_embedding_size * num_kernels) ->
        # batch_size x max_sequence_length x (word_embedding_size * num_kernels)
        char2word_dense = self.char2word_dense(self.dense_dropout(char2word_embedded))
        char2word_residual = self.residual_add([char2word_dense, char2word_embedded])

        if self.use_words:
            # batch_size x max_sequence_length x (word_embedding_size * (num_kernels + 1))
            embedded = self.concat_embeddings([char2word_residual, self.word_input_dropout(self.word_embed(inp))])
        else:
            embedded = char2word_residual

        # batch_size x max_sequence_length x (word_embedding_size * (num_kernels + 1)) ->
        # batch_size x encoded_size

        encoded_sequence = self.encoder(embedded)
        encoded = self.masked_global_max_pool(encoded_sequence)
        return encoded


class SimilarityModel(KerasModel):
    def __init__(self, vocab_size, charset_size, num_negative_samples, gamma=5, use_words=True,
                 max_sequence_length=35, max_word_length=40, char_kernel_sizes=(2, 3), word_embedding_size=64,
                 char_embedding_size=32, dropout=0.2, recurrent_dropout=0.2, encoder_hidden_units=64,
                 l2_regularization=0.01, bidirectional=True, ):
        super(SimilarityModel, self).__init__()
        self.encoder = DeepEncoder(vocab_size=vocab_size,
                                   charset_size=charset_size,
                                   max_sequence_length=max_sequence_length,
                                   max_word_length=max_word_length,
                                   use_words=use_words,
                                   char_kernel_sizes=char_kernel_sizes,
                                   word_embedding_size=word_embedding_size,
                                   char_embedding_size=char_embedding_size,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   encoder_hidden_units=encoder_hidden_units,
                                   l2_regularization=l2_regularization,
                                   bidirectional=bidirectional)

        # --- Input ---
        self.query_word_inputs = Input(
            shape=(max_sequence_length,),
            dtype='int32',
            name='input_word_indices'
        )

        self.query_char_inputs = Input(
            shape=(max_sequence_length, max_word_length),
            dtype='int32',
            name='input_char_indices'
        )

        self.documents_word_inputs = Input(
            shape=(max_sequence_length,),
            dtype='int32',
            name='doc_word_indices'
        )
        self.documents_char_inputs = Input(
            shape=(max_sequence_length, max_word_length),
            dtype='int32',
            name='doc_char_indices'
        )

        self.num_negative_samples = num_negative_samples
        self.gamma = gamma

        self.permute_as_021 = Permute(
            dims=(2, 1)
        )

        self.cosine_similarity = Lambda(
            function=self.compute_similarities,
            name='compute_sims'
        )

    def compute_similarities(self, packed_input):
        query, documents = packed_input
        documents = K.reshape(documents, shape=(-1, self.num_negative_samples + 1, K.int_shape(documents)[-1]))
        nums = K.dot(query, self.permute_as_021(documents))
        query_norm = K.sqrt(K.sum(K.square(query), axis=1, keepdims=True))
        documents_norm = K.sqrt(K.sum(K.square(documents), axis=2, keepdims=True))
        dens = K.dot(query_norm, self.permute_as_021(documents_norm)) + epsilon
        scores = nums / dens
        return scores

    def compile_model(self, optimizer='adam'):
        def _loss(y, y_pred):
            diffs = K.gather(reference=y_pred, indices=K.cast(y, dtype='int32')) - y_pred
            return K.logsumexp(-self.gamma * diffs)

        query_encoded = self.encoder.encode(self.query_word_inputs, self.query_char_inputs)
        document_encoded = self.encoder.encode(self.documents_word_inputs, self.documents_char_inputs)
        scores = self.cosine_similarity([query_encoded, document_encoded])
        self.model = Model(
            inputs=[self.query_word_inputs, self.query_char_inputs,
                    self.documents_word_inputs, self.documents_char_inputs],
            outputs=scores
        )
        self.model.compile(optimizer=optimizer, loss=_loss)


class ClassifierModel(KerasModel):
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'

    def __init__(self, vocab_size, charset_size, num_classes, mode, use_words=True,
                 max_sequence_length=35, max_word_length=40, char_kernel_sizes=(2, 3), word_embedding_size=64,
                 char_embedding_size=32, dropout=0.2, recurrent_dropout=0.2, encoder_hidden_units=64,
                 l2_regularization=0.01, bidirectional=True):
        super(ClassifierModel, self).__init__()
        self.encoder = DeepEncoder(vocab_size=vocab_size,
                                   charset_size=charset_size,
                                   max_sequence_length=max_sequence_length,
                                   max_word_length=max_word_length,
                                   use_words=use_words,
                                   char_kernel_sizes=char_kernel_sizes,
                                   word_embedding_size=word_embedding_size,
                                   char_embedding_size=char_embedding_size,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   encoder_hidden_units=encoder_hidden_units,
                                   l2_regularization=l2_regularization,
                                   bidirectional=bidirectional)

        # --- Input ---
        self.query_word_inputs = Input(
            shape=(max_sequence_length,),
            dtype='int32',
            name='input_word_indices'
        )

        self.query_char_inputs = Input(
            shape=(max_sequence_length, max_word_length),
            dtype='int32',
            name='input_char_indices'
        )

        self.mode = mode
        self.num_classes = num_classes
        self.dense_out = Dense(
            units=self.num_classes,
            activation='softmax' if mode == ClassifierModel.MULTICLASS else 'sigmoid',
            kernel_regularizer=regularizers.l2(l2_regularization)
        )
        self.dense_dropout = Dropout(rate=dropout, seed=model_seed)

    def compile_model(self, optimizer='adam'):
        encoded = self.encoder.encode(self.query_word_inputs, self.query_char_inputs)
        y_pred = self.dense_out(self.dense_dropout(encoded))
        loss = 'categorical_crossentropy' if self.mode == ClassifierModel.MULTICLASS else 'binary_crossentropy'
        self.model = Model(inputs=[self.query_word_inputs, self.query_char_inputs], outputs=y_pred)
        self.model.compile(optimizer=optimizer, loss=loss)

