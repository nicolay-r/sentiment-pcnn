# Sentiment Piecewise Convolutional Neural Network
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/TensorFlow-1.4.1-yellowgreen.svg)

This project represents an implementation of PCNN [[zeng2015distant](http://www.aclweb.org/anthology/D15-1203)], dubbed as
*Piecewise Convolutional Neural Network*, written in Tensorflow.
Considered as an application for sentiment attitudes extraction task.

![alt text](docs/pcnn.png)

Architectures implementation:
* [[cnn](networks/context/architectures/cnn.py)]
* [[pcnn](networks/context/architectures/pcnn.py)]

Model configuration parameters:
* [[base](networks/context/configurations/base.py)]
* [[cnn/pcnn](networks/context/configurations/cnn.py)]

The ```master``` branch represents the latest implementation.
Experiment details presented in other branches.
The historical changeset is as follows:
1. 2018 -- Original implementation, classification is strongly by contexts:
    * **DAMDID-2018** Conference [[paper](http://ceur-ws.org/Vol-2277/paper33.pdf)] 
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/damdid-2018)]
    * **RUSSIR-2018** Conference (Not published)
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/russir-2018)]
    * **CLLS-2018** Conference (Were not published, includes presentation slides)
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/clls-2018)]
1. 2019 -- Implementation has been significantly refactored. Application of aggregation function of related contexts with different functions, especially with RNN application:
    * **CCIS-1003** Journal
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/ccis-2019)]


Dataset
-------
We use [RuSentRel 1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/)
corpus consisted of analytical articles from Internet-portal
inosmi.ru.

Installation
------------

Using [virtualenv](https://www.pythoncentral.io/how-to-install-virtualenv-python/).
Create virtual environment, suppose `my_env`, and activate it as follows:
```
virtualenv my_env
source my_env/bin/activate
```

Use `Makefile` to install
[re-core v0.19.1](https://github.com/nicolay-r/sentiment-erc-core/tree/release_19_1) library and download
[RuSentRel-v1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/):
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
And we are ready to apply model (CNN) with different settings by simply running:
```
./predict/context/cnn.py
```
or PCNN as follows:
```
./predict/context/pcnn.py
```

To apply neural network to get aggregated results, use the following script next,
which evaluates result in case of different cell types of rnn modes (RNN, GRU, LSTM):
```
./predict/text/rnn_all.py
```

Related works
-------------
* Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao, Distant
supervision for relation extraction via piecewise convolutional
neural networks, Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing, 2015,
pp. 1753–1762
[[paper](http://www.aclweb.org/anthology/D15-1203)]
