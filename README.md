# sentiment-pcnn

PCNN stands for Piecewise Convolutional Neural Network, written in
Tensorflow.
Considered as an application for sentiment attitudes extraction task.

[Slides](docs/slides.pdf)

![alt text][docs/pcnn.png]

Dataset
-------
We use [RuSentRel](https://github.com/nicolay-r/RuSentRel)
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
[core](https://github.com/nicolay-r/sentiment-erc-core) library and download
[dataset](https://github.com/nicolay-r/RuSentRel):
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
./predict_pcnn.py
```
Or a version which implements a cross validation:
```
./predict_pcnn_cv.py
```

References
----------

Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao, Distant
supervision for relation extraction via piecewise convolutional
neural networks, Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing, 2015,
pp. 1753–1762
