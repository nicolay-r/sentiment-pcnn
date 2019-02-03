#!/usr/bin/python
import tensorflow as tf
from tensorflow.python.framework import dtypes
from networks.context.architectures.cnn import VanillaCNN
from networks.context.configurations.cnn import CNNConfig
from networks.context.processing.sample import Sample
import utils


class PiecewiseCNN(VanillaCNN):

    def __init__(self, config):
        assert(isinstance(config, CNNConfig))
        self.cfg = config

        tf.reset_default_graph()

        nlp_vector_size = 0 if not self.cfg.UseNLPVector else self.cfg.NLPVectorSize

        # Hidden state variables
        W = tf.Variable(tf.random_normal([3 * self.cfg.FiltersCount + nlp_vector_size, self.cfg.ClassesCount]), dtype=tf.float32)
        bias = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)
        W2 = tf.Variable(tf.random_normal([self.cfg.HiddenSize, self.cfg.ClassesCount]), dtype=tf.float32)
        bias2 = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)
        conv_filter = tf.Variable(tf.random_normal([self.cfg.WindowSize * self.get_embedding_width(self.cfg), 1, self.cfg.FiltersCount]), dtype=tf.float32)

        # Input placeholders
        self.x = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize])
        self.p_subj_dist = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.p_obj_dist = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])
        self.p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize])  # left indices for each batch
        self.p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize])  # right indices for each batch
        self.nlp_features = tf.placeholder(dtype=tf.float32, shape=[self.cfg.BatchSize, nlp_vector_size])
        self.pos = tf.placeholder(tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext])

        embedded_terms = utils.init_embedded_terms(
            x=self.x,
            dist_from_obj=self.dist_from_obj,
            dist_from_subj=self.dist_from_subj,
            pos=self.pos,
            cfg=self.cfg)

        embedded_terms = self.padding(embedded_terms, self.cfg.WindowSize)

        # concatenate rows of matrix
        bwc_line = tf.reshape(embedded_terms, [self.cfg.BatchSize,
                                               (self.cfg.TermsPerContext + (self.cfg.WindowSize - 1)) * self.get_embedding_width(self.cfg),
                                               1])
        bwc_conv = tf.nn.conv1d(bwc_line, conv_filter, self.get_embedding_width(self.cfg),
                                "VALID",
                                data_format="NHWC",
                                name="conv")

        # slice all data into 3 parts -- before, inner, and after according to relation
        sliced = tf.TensorArray(dtype=tf.float32, size=self.cfg.BatchSize, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
                lambda i, *_: tf.less(i, self.cfg.BatchSize),
                self.splitting,
                [0, self.p_subj_ind, self.p_obj_ind, bwc_conv, self.cfg.FiltersCount, sliced])
        sliced = tf.squeeze(sliced.concat())
        embedded_terms = utils.init_embedded_terms(
            x=self.x,
            dist_from_obj=self.dist_from_obj,
            dist_from_subj=self.dist_from_subj,
            pos=self.pos,
            cfg=self.cfg)
        # maxpool
        bwgc_mpool = tf.nn.max_pool(
                sliced,
                [1, 1, self.cfg.TermsPerContext, 1],
                [1, 1, self.cfg.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bwc_mpool = tf.squeeze(bwgc_mpool, [2])
        bcw_mpool = tf.transpose(bwc_mpool, perm=[0, 2, 1])
        bc_pmpool = tf.reshape(bcw_mpool, [self.cfg.BatchSize, 3 * self.cfg.FiltersCount])

        tensor_to_activate = bc_pmpool
        if self.cfg.UseNLPVector:
            tensor_to_activate = tf.concat([bc_pmpool, self.nlp_features], 1)

        g = tf.tanh(tensor_to_activate)
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
            self.p_subj_dist: input[Sample.I_SUBJ_DISTS],
            self.p_obj_dist: input[Sample.I_OBJ_DISTS],
            self.p_subj_ind: input[Sample.I_SUBJ_IND],
            self.p_obj_ind: input[Sample.I_OBJ_IND],
        }

        if self.cfg.UsePOSEmbedding:
            feed_dict[self.pos] = input[Sample.I_POS_INDS]

        if self.cfg.UseNLPVector:
            feed_dict[self.nlp_features] = input[Sample.I_NLP_FEATURES]

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
    def Accuracy(self):
        return self.accuracy

    @property
    def Labels(self):
        return self.labels

    @property
    def Output(self):
        return self.output

    @property
    def ParametersDictionary(self):
        return {}
