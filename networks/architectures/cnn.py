import tensorflow as tf
from networks.configurations.cnn import CNNConfig
from networks.architectures.base import NeuralNetwork
from core.env import stemmer
import utils


class VanillaCNN(NeuralNetwork):

    def __init__(self, config, embedding_shape):
        assert(isinstance(config, CNNConfig))
        self.cfg = config

        tf.reset_default_graph()

        batch_size = self.cfg.bag_size * self.cfg.bags_per_minibatch
        embedding_size_p = embedding_shape[1] + 2 + (self.cfg.pos_emb_size if self.cfg.use_pos else 0)

        # Hidden state variables
        W = tf.Variable(tf.random_normal([self.cfg.filters_count, self.cfg.output_classes]), dtype=tf.float32)
        bias = tf.Variable(tf.random_normal([self.cfg.output_classes]), dtype=tf.float32)
        conv_filter = tf.Variable(tf.random_normal([self.cfg.window_size * embedding_size_p, 1, self.cfg.filters_count]), dtype=tf.float32)

        # Input placeholders
        self.x = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.cfg.words_per_news])
        self.dist_from_subj = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.cfg.words_per_news])
        self.dist_from_obj = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.cfg.words_per_news])
        self.y = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        self.E = tf.placeholder(dtype=tf.float32, shape=embedding_shape)
        bernoulli = tf.distributions.Bernoulli(self.cfg.dropout, dtype=tf.float32)
        # Part of Speech parameters
        self.pos = tf.placeholder(tf.int32, shape=[batch_size, self.cfg.words_per_news])
        self.pos_emb = tf.get_variable(dtype=tf.float32, shape=[len(stemmer.pos_names), self.cfg.pos_emb_size], trainable=True, name="pos_emb")

        # Apply embeding for input x indices
        emb_words = tf.nn.embedding_lookup(self.E, self.x)

        # Part of speech embedding
        if self.cfg.use_pos:
            emb_pos = tf.nn.embedding_lookup(self.pos_emb, self.pos)
            emb_words = tf.concat([emb_words, emb_pos], axis=-1)

        emb_words = utils.merge_with_embedding(emb_words, [self.dist_from_subj, self.dist_from_obj])

        # Add padding embedding rows (reason -- have the same amount of rows after conv1d).
        left_padding = (self.cfg.window_size - 1) / 2
        right_padding = (self.cfg.window_size - 1) - left_padding
        emb_words = tf.pad(emb_words, [[0, 0], [left_padding, right_padding], [0, 0]])

        # Concatenate rows of matrix
        bwc_line = tf.reshape(emb_words, [batch_size, (self.cfg.words_per_news + (self.cfg.window_size - 1)) * embedding_size_p, 1])
        bwc_conv = tf.nn.conv1d(bwc_line, conv_filter, embedding_size_p, "VALID", data_format="NHWC", name="conv")
        bwgc_conv = tf.reshape(bwc_conv, [batch_size, 1, self.cfg.words_per_news, self.cfg.filters_count])

        # Maxpool
        bwgc_mpool = tf.nn.max_pool(
                bwgc_conv,
                [1, 1, self.cfg.words_per_news, 1],
                [1, 1, self.cfg.words_per_news, 1],
                padding='VALID',
                data_format="NHWC")

        bc_mpool = tf.squeeze(bwgc_mpool, axis=[1, 2])
        bc_pmpool = tf.reshape(bc_mpool, [batch_size, self.cfg.filters_count])
        g = tf.tanh(bc_pmpool)

        self.labels = tf.argmax(self.get_logits(g, W, bias), axis=1)

        if self.cfg.use_bernoulli_mask:
            # Apply Bernoulli mask for 'g'
            r = bernoulli.sample(sample_shape=[1, 3*self.cfg.filters_count])
            r_batch = tf.matmul(tf.constant(1, shape=[batch_size, 1], dtype=tf.float32), r)
            g = tf.multiply(g, r_batch)

        logits_unscaled_dropout = tf.nn.dropout(
            self.get_logits(g, W, bias),
            self.cfg.dropout)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_unscaled_dropout,
                labels=self.y)
        cross_entropy_per_bag = tf.reshape(cross_entropy, [self.cfg.bags_per_minibatch, self.cfg.bag_size])

        self.cost = tf.reduce_max(cross_entropy_per_bag, axis=1)

    def get_feed_dict(self, X, labels, dist_from_subj, dist_from_obj, embedding, pos=None):
        feed_dict = {
            self.x: X,
            self.y: labels,
            self.dist_from_subj: dist_from_subj,
            self.dist_from_obj: dist_from_obj,
            self.E: embedding
        }

        if self.cfg.use_pos:
            feed_dict[self.pos] = pos

        return feed_dict

    @staticmethod
    def get_logits(g, W, bias):
        return tf.matmul(g, W) + bias

    @property
    def Cost(self):
        return self.cost

    @property
    def Labels(self):
        return self.labels

    @property
    def ParametersDictionary(self):
        return self.cfg.get_paramters()
