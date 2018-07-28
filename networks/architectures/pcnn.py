#!/usr/bin/python
import tensorflow as tf
from networks.architectures.base import NeuralNetwork
from networks.configurations.cnn import CNNConfig
from core.env import stemmer
import utils


class PiecewiseCNN(NeuralNetwork):

    def __init__(self, config, embedding_shape):
        assert(isinstance(config, CNNConfig))
        self.cfg = config

        tf.reset_default_graph()

        nlp_vector_size = 0 if not self.cfg.use_nlp_vector else self.cfg.nlp_vector_size
        batch_size = self.cfg.bag_size * self.cfg.bags_per_minibatch
        total_embedding_size = embedding_shape[1] + 2 + (self.cfg.pos_emb_size if self.cfg.use_pos else 0) # embedding + dp_1 + dp_2 + pos

        # Hidden state variables
        W = tf.Variable(tf.random_normal([3 * self.cfg.filters_count + nlp_vector_size, self.cfg.output_classes]), dtype=tf.float32)
        bias = tf.Variable(tf.random_normal([self.cfg.output_classes]), dtype=tf.float32)
        conv_filter = tf.Variable(tf.random_normal([self.cfg.window_size * total_embedding_size, 1, self.cfg.filters_count]), dtype=tf.float32)

        # Input placeholders
        self.x = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.cfg.words_per_news])
        self.p_subj_dist = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.cfg.words_per_news])
        self.p_obj_dist = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.cfg.words_per_news])
        self.p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size])  # left indices for each batch
        self.p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size])  # right indices for each batch
        self.nlp_features = tf.placeholder(dtype=tf.float32, shape=[batch_size, nlp_vector_size])
        self.y = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        self.E = tf.placeholder(dtype=tf.float32, shape=embedding_shape)
        bernoulli = tf.distributions.Bernoulli(self.cfg.dropout, dtype=tf.float32)
        # Part of Speech parameters
        self.pos = tf.placeholder(tf.int32, shape=[batch_size, self.cfg.words_per_news])
        self.pos_emb = tf.get_variable(dtype=tf.float32, shape=[len(stemmer.pos_names), self.cfg.pos_emb_size], trainable=True, name="pos_emb")

        # apply embeding for input x indices
        word_emb = tf.nn.embedding_lookup(self.E, self.x)

        # Part of speech embedding
        if self.cfg.use_pos:
            emb_pos = tf.nn.embedding_lookup(self.pos_emb, self.pos)
            word_emb = tf.concat([word_emb, emb_pos], axis=-1)

        word_emb = utils.merge_with_embedding(word_emb, [self.p_subj_dist, self.p_obj_dist])

        # add padding embedding rows (reason -- have the same amount of rows after conv1d).
        left_padding = (self.cfg.window_size - 1) / 2
        right_padding = (self.cfg.window_size - 1) - left_padding
        word_emb = tf.pad(word_emb, [[0, 0], [left_padding, right_padding], [0, 0]])

        # concatenate rows of matrix
        bwc_line = tf.reshape(word_emb, [batch_size, (self.cfg.words_per_news + (self.cfg.window_size - 1)) * total_embedding_size, 1])
        bwc_conv = tf.nn.conv1d(bwc_line, conv_filter, total_embedding_size, "VALID", data_format="NHWC", name="conv")

        # slice all data into 3 parts -- before, inner, and after according to relation
        sliced = tf.TensorArray(dtype=tf.float32, size=batch_size, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
                lambda i, *_: tf.less(i, batch_size),
                self.splitting,
                [0, self.p_subj_ind, self.p_obj_ind, bwc_conv, self.cfg.filters_count, sliced])
        sliced = tf.squeeze(sliced.concat())

        # maxpool
        bwgc_mpool = tf.nn.max_pool(
                sliced,
                [1, 1, self.cfg.words_per_news, 1],
                [1, 1, self.cfg.words_per_news, 1],
                padding='VALID',
                data_format="NHWC")

        bwc_mpool = tf.squeeze(bwgc_mpool, [2])
        bcw_mpool = tf.transpose(bwc_mpool, perm=[0, 2, 1])
        bc_pmpool = tf.reshape(bcw_mpool, [batch_size, 3*self.cfg.filters_count])

        tensor_to_activate = bc_pmpool
        if self.cfg.use_nlp_vector:
            tensor_to_activate = tf.concat([bc_pmpool, self.nlp_features], 1)

        g = tf.tanh(tensor_to_activate)
        logits_unscaled = tf.matmul(g, W) + bias
        self.labels = tf.argmax(tf.nn.softmax(logits_unscaled), axis=1)

        if self.cfg.use_bernoulli_mask:
            # apply Bernoulli mask for 'g'
            r = bernoulli.sample(sample_shape=[1, 3*self.cfg.filters_count])
            r_batch = tf.matmul(tf.constant(1, shape=[batch_size, 1], dtype=tf.float32), r)
            masked_g = tf.multiply(g, r_batch)
            logits_unscaled = tf.matmul(masked_g, W) + bias

        logits_unscaled_dropout = tf.nn.dropout(logits_unscaled, self.cfg.dropout)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_unscaled_dropout,
                labels=self.y)
        cross_entropy_per_bag = tf.reshape(cross_entropy,
                                           [self.cfg.bags_per_minibatch, self.cfg.bag_size])

        self.cost = tf.reduce_max(cross_entropy_per_bag, axis=1)

    def get_feed_dict(self, X, labels,
                      p_subj_dist, p_obj_dist,
                      p_subj_ind, p_obj_ind,
                      embedding, pos=None, nlp_features=None):
        feed_dict = {
            self.x: X,
            self.y: labels,
            self.p_subj_dist: p_subj_dist,
            self.p_obj_dist: p_obj_dist,
            self.p_subj_ind: p_subj_ind,
            self.p_obj_ind: p_obj_ind,
            self.E: embedding
        }

        if self.cfg.use_pos:
            feed_dict[self.pos] = pos

        if self.cfg.use_nlp_vector:
            feed_dict[self.nlp_features] = nlp_features

        return feed_dict

    @staticmethod
    def splitting(i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs):
        l_ind = tf.minimum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # left
        r_ind = tf.maximum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # right

        w = tf.Variable(bwc_conv.shape[1], dtype=tf.int32) # total width (words count)

        b_slice_from = [i, 0, 0]
        b_slice_size = tf.concat([[1], l_ind, [channels_count]], 0)
        m_slice_from = tf.concat([[i], l_ind, [0]], 0)
        m_slice_size = tf.concat([[1], r_ind - l_ind, [channels_count]], 0)
        a_slice_from = tf.concat([[i], r_ind, [0]], 0)
        a_slice_size = tf.concat([[1], w-r_ind, [channels_count]], 0)

        bwc_split_b = tf.slice(bwc_conv, b_slice_from, b_slice_size)
        bwc_split_m = tf.slice(bwc_conv, m_slice_from, m_slice_size)
        bwc_split_a = tf.slice(bwc_conv, a_slice_from, a_slice_size)

        pad_b = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_m = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-r_ind+l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_a = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([r_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=tf.float32.min)
        bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=tf.float32.min)
        bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=tf.float32.min)

        outputs = outputs.write(i, [[bwc_split_b, bwc_split_m, bwc_split_a]])

        i += 1
        return i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs

    @property
    def Cost(self):
        return self.cost

    @property
    def Labels(self):
        return self.labels

    @property
    def ParametersDictionary(self):
        return self.cfg.get_paramters()
