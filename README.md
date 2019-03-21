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

Changeset
---------
The ```master``` branch represents the latest implementation.
Experiment details presented in other branches.
The historical changeset is as follows:
* 2018 -- Original implementation, classification is strongly by contexts:
    * **DAMDID-2018** Conference
        [[paper](http://ceur-ws.org/Vol-2277/paper33.pdf)] 
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/damdid-2018)]
    * **RUSSIR-2018** Conference (Not published) 
        [[poster](https://github.com/nicolay-r/sentiment-pcnn/blob/russir-2018/docs/poster.pdf)]
        [[paper](https://github.com/nicolay-r/sentiment-pcnn/blob/russir-2018/docs/paper.pdf)]
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/russir-2018)]
    * **CLLS-2018** Conference
        [[paper](https://doi.org/10.29007/26g7)]
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/clls-2018)]
* 2019 -- Implementation has been significantly refactored. Application of aggregation function of related contexts with different functions, especially with RNN application:
    * **CCIS** (vol. 1003) Journal
        [[code](https://github.com/nicolay-r/sentiment-pcnn/tree/ccis-2019)]


Dataset
-------
We use [RuSentRel 1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/)
corpus consisted of analytical articles from Internet-portal
inosmi.ru.

Related works
-------------
* Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao, Distant
supervision for relation extraction via piecewise convolutional
neural networks, Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing, 2015,
pp. 1753–1762
[[paper](http://www.aclweb.org/anthology/D15-1203)]
