import tensorflow as tf
from tensorflow.python.framework import dtypes
from networks.context.configurations.cnn import CNNConfig
from networks.network import NeuralNetwork
from networks.context.processing.sample import Sample

import utils


class VanillaCNN(NeuralNetwork):

    def __init__(self, config):
        assert(isinstance(config, CNNConfig))
        self.cfg = config

        tf.reset_default_graph()

        # Hidden state variables
        W = tf.Variable(tf.random_normal([self.cfg.FiltersCount, self.cfg.HiddenSize]), dtype=tf.float32)
        bias = tf.Variable(tf.random_normal([self.cfg.HiddenSize]), dtype=tf.float32)
        W2 = tf.Variable(tf.random_normal([self.cfg.HiddenSize, self.cfg.ClassesCount]), dtype=tf.float32)
        bias2 = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)
        conv_filter = tf.Variable(tf.random_normal([self.cfg.WindowSize * self.get_embedding_width(self.cfg), 1, self.cfg.FiltersCount]), dtype=tf.float32)

        # Input placeholders
        self.x = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize])
        self.dist_from_subj = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.dist_from_obj = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.pos = tf.placeholder(tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])

        embedded_terms = utils.init_embedded_terms(
            x=self.x,
            dist_from_obj=self.dist_from_obj,
            dist_from_subj=self.dist_from_subj,
            pos=self.pos,
            cfg=self.cfg)

        embedded_terms = self.padding(embedded_terms, self.cfg.WindowSize)

        # Concatenate rows of matrix
        bwc_line = tf.reshape(embedded_terms,
                              [self.cfg.BatchSize,
                               (self.cfg.TermsPerContext + (self.cfg.WindowSize - 1)) * self.get_embedding_width(self.cfg),
                               1])

        bwc_conv = tf.nn.conv1d(bwc_line, conv_filter, self.get_embedding_width(self.cfg),
                                "VALID",
                                data_format="NHWC",
                                name="conv")

        bwgc_conv = tf.reshape(bwc_conv, [self.cfg.BatchSize,
                                          1,
                                          self.cfg.TermsPerContext,
                                          self.cfg.FiltersCount])

        # Maxpool
        bwgc_mpool = tf.nn.max_pool(
                bwgc_conv,
                [1, 1, self.cfg.TermsPerContext, 1],
                [1, 1, self.cfg.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bc_mpool = tf.squeeze(bwgc_mpool, axis=[1, 2])
        bc_pmpool = tf.reshape(bc_mpool, [self.cfg.BatchSize, self.cfg.FiltersCount])
        g = tf.tanh(bc_pmpool)

        logits_unscaled = utils.get_two_layer_logits(g, W, bias, W2, bias2)

        self.output = tf.nn.softmax(logits_unscaled)
        self.labels = tf.argmax(self.output, axis=1, output_type=dtypes.int32)

        if self.cfg.UseBernoulliMask:
            masked_g = self.init_masked_g(g, self.cfg)
            logits_unscaled = utils.get_two_layer_logits(masked_g, W, bias, W2, bias2)

        self.weights, self.cost = utils.init_weighted_cost(
            logits_unscaled_dropout=tf.nn.dropout(logits_unscaled, self.cfg.Dropout),
            true_labels=self.y,
            cfg=self.cfg)

        self.accuracy = utils.init_accuracy(labels=self.labels,
                                            true_labels=self.y)

    def create_feed_dict(self, input, data_type):

        feed_dict = {
            self.x: input[Sample.I_X_INDS],
            self.y: input[Sample.I_LABELS],
            self.dist_from_subj: input[Sample.I_SUBJ_DISTS],
            self.dist_from_obj: input[Sample.I_OBJ_DISTS],
        }

        if self.cfg.UsePOSEmbedding:
            feed_dict[self.pos] = input[Sample.I_POS_INDS]

        return feed_dict

    @staticmethod
    def get_embedding_width(cfg):
        assert(isinstance(cfg, CNNConfig))
        return cfg.TermEmbeddingShape[1] + \
               2 * cfg.DistanceEmbeddingSize + \
               (cfg.PosEmbeddingSize if cfg.UsePOSEmbedding else 0)

    @staticmethod
    def padding(embedded_data, window_size):
        assert(isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])

    @staticmethod
    def init_masked_g(g, cfg):
        assert(isinstance(cfg, CNNConfig))
        bernoulli = tf.distributions.Bernoulli(cfg.Dropout, dtype=tf.float32)
        r = bernoulli.sample(sample_shape=[1, 3 * cfg.FiltersCount])
        r_batch = tf.matmul(tf.constant(1, shape=[cfg.BatchSize, 1], dtype=tf.float32), r)
        return tf.multiply(g, r_batch)

    @property
    def Accuracy(self):
        return self.accuracy

    @property
    def Cost(self):
        return self.cost

    @property
    def Labels(self):
        return self.labels

    @property
    def Output(self):
        return self.output

    @property
    def Log(self):
        params = [#("output", self.output),
                  ("predictions", self.labels)]
        return [list(p) for p in zip(*params)]

    @property
    def ParametersDictionary(self):
        return {}
