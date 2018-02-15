### Deep Character CNN LSTM Encoder with Classification and Similarity Models
In Keras

Overall Idea:

- Convolve over character embeddings with different kernel sizes
- Concat them to get the char-word embedding
- Pass them through a Dense layer with Residual connection
- Optionally concat them with separate word embedding
- Pass sequence of obtained word embeddings through a LSTM encoder
- Train with a constrastive loss function (see References)

### Work in Progress

- TODO: Add loading utils
- TODO: Add preprocessing and padding utils
- TODO: Add batching utils
- TODO: Add model training code
- TODO: Add model continue-training code
- TODO: Test Similarity implementation on Quora similar pair dataset
- TODO: Test Classification implementation on Kaggle Toxic internet comments dataset
- TODO: Tune Hyperparameters and try different modifications to architectures
- TODO: Take Hyperparameters using argparse
- TODO: Add tensorboard and tfdbg support

### Example Usage:

```python
from model import ClassifierModel, SimilarityModel

classifier = ClassifierModel(vocab_size=10000,
                             charset_size=100,
                             num_classes=5,
                             mode=ClassifierModel.MULTILABEL,
                             char_kernel_sizes=(3,),
                             encoder_hidden_units=128,
                             bidirectional=False)
classifier.compile_model()

similarity_model = SimilarityModel(vocab_size=10000,
                                   charset_size=100,
                                   num_negative_samples=1)
similarity_model.compile_model()
```

### References:


**Overall Idea**

1. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)][1]


**Encoder architecture heavily inspired from**
1. [Character-Aware Neural Language Models (2015), Kim et. al.][2]
2. [dpressel/baseline][3]

**Loss function taken from**
1. [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval (2014)][4]


**Other Contrastive Loss functions to try**
1. [StarSpace: Embed All The Things! (2017) Wu et. al.][5]
2. [Comparision of loss functions for deep embedding][6]


[1]: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195
[2]: https://arxiv.org/abs/1508.06615
[3]: https://github.com/dpressel/baseline/tree/master/python
[4]: https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/
[5]: https://arxiv.org/abs/1709.03856
[6]: https://www.slideshare.net/CenkBircanolu/a-comparison-of-loss-function-on-deep-embedding