# Sentiment Piecewise Convolutional Neural Network
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/TensorFlow-1.4.1-yellowgreen.svg)

This project represents an implementation of PCNN [[zeng2015distant](http://www.aclweb.org/anthology/D15-1203)], dubbed as
*Piecewise Convolutional Neural Network*, written in Tensorflow.
Considered as an application for sentiment attitudes extraction task.

Architectures implementation:
* [[cnn](networks/architectures/cnn.py)]
* [[pcnn](networks/architectures/pcnn.py)]

For more details, see (or [References](#references) section):

1. CEUR Workshop Proceedings [paper](docs/clls-2018-hse.pdf)

2. CLLS-2018, Presentation [slides](docs/slides.pdf)


![alt text](docs/pcnn.png)

Dataset
-------
We use [RuSentRel 1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/)
corpus consisted of analytical articles from Internet-portal
inosmi.ru.


This collection provides an attitudes for whole articles. 
Therefore it is necessary to find a **context**(s) related to each attitude -- 
a part of text, which includes Subject and Object.

For **sentiment attitudes** (about 10%) we kept all sentences that has related object and subject, and assume such contexts with the related attitude label.
> We kept all sentences (instead of the first appeared) because the lack of sentiment contexts in comparison with neutral one.

For **neutral attitudes** (about 90%) we kept only first appeared sentence that has related object and subject.


Results
-------
Model [[configuration](networks/configurations/cnn.py)].
Table below illustrates CNN/PCNN results in comparison with **baselines**
(neg, pos, distr), and **classifiers** (KNN, SVM, Random Forest)
based on handcrafted NLP features.
Last row represent an asessment of agreement between two annotators.

| Model               | Precision | Recall | F1(P,N)  |
|--------------------:|:---------:|:------:|:--------:|
|Baseline neg         |  0.03     | 0.39   | 0.05     |
|Baseline pos         |  0.02     | 0.40   | 0.04     |
|Baseline distr       |  0.05     | 0.23   | 0.08     |
|KNN                  |  0.18     | 0.06   | 0.09     |
|SVM (GRID)           |  0.09     | 0.36   | 0.15     |
|Random forest (GRID) |  0.41     | 0.21   | 0.27     |
| **CNN**  		      |  0.41     | 0.23   | **0.31** |
| **PCNN** 		      |  0.42     | 0.23   | **0.31** |
|Expert agreement     |  0.62     | 0.49   | 0.55     |

Comparison of CNN and PCNN during training process illustrated in figure below:

![alt text](docs/f1_and_cost.png)
> Using default train/test separation of RuSentRel v1.0 collection; filters=200;
window\_size=3; **left subfigure**: F1(P, N) reults per epoch fot test subset; 
**right subfigure**: cost values per epoch;
using piecewise cnn results in training speed, and latter reach better results 
faster than vanilla cnn.

> **NOTE:** For cost evaluation, we use `tf.nn.sparse_softmax_cross_entropy_with_logits()` which 
compares results by an exact class, not by distribution; **class weights** were not taken
into account.

Installation
------------

Using [virtualenv](https://www.pythoncentral.io/how-to-install-virtualenv-python/).
Create virtual environment, suppose `my_env`, and activate it as follows:
```
virtualenv my_env
source my_env/bin/activate
```

Use `Makefile` to install
[core](https://github.com/nicolay-r/sentiment-erc-core) library and download
[dataset](https://github.com/nicolay-r/RuSentRel/tree/v1.0/):
```
make install
```

We use word2vec
[model](http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz)
which were taken from rusvectores.org and used for an embedding layer completion:
```
make download_model
```

Usage
-----
The dataset provides only sentiment attitudes.
For extraction of positive and negative attitudes we additionally introduce
(extract from news) **neutral attudes** to distinguish really sentiment
attitudes from neutral one.

At first, we compose a list of neutral relations per each article by running:
```
./neutrals.py
```
And we are ready to apply model with different settings by simply rinning:
```
./predict_cnn.py
```

References
----------

<a name="references"></a>
```
@article{rusnachenko2018piecewisecnn,
    Author = {Rusnachenko, N. and Loukachevitch, N.},
    Title = {Using Convolutional Neural Networks for Sentiment Attitude
             Extraction from Analytical Texts},
    Journal = {In Proceedings of CEUR Workshop, CLLS-2018 Conference},
    url = {ceur-ws.org},
    Year = {2018}
}
```

Related works
-------------

* Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao, Distant
supervision for relation extraction via piecewise convolutional
neural networks, Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing, 2015,
pp. 1753–1762
[[paper](http://www.aclweb.org/anthology/D15-1203)]
