import tensorflow as tf
from tensorflow.python.framework import dtypes
from networks.io import DataType
from networks.text.processing.batch import Batch
from networks.text.configurations.rnn import RNNConfig
from networks.network import NeuralNetwork


class RNN(NeuralNetwork):

    def __init__(self, config):
        assert(isinstance(config, RNNConfig))
        self.cfg = config

        # https://stackoverflow.com/questions/47296969/valueerror-variable-rnn-basic-rnn-cell-kernel-already-exists-disallowed-did-y
        tf.reset_default_graph()

        # Input parameters
        self.x = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.GroupSize])
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize])
        self.emb_contexts = tf.placeholder(dtype=tf.float32, shape=self.cfg.EmbeddingShape)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.sentences_inds = tf.placeholder(dtype=tf.float32, shape=[self.cfg.EmbeddingShape[0],
                                                                      self.cfg.TextPartsCount])

        embedding = self.emb_contexts
        if self.cfg.UseSentenceIndices:
            embedding = tf.concat([self.emb_contexts, self.sentences_inds], axis=-1)

        embedded_contexts = tf.nn.embedding_lookup(embedding, self.x)

        with tf.name_scope("rnn"):
            sequence_length = self._length(self.x)
            cell = self._get_cell(self.cfg.HiddenSize, self.cfg.CellType)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=embedded_contexts,
                                               sequence_length=sequence_length,
                                               dtype=tf.float32)
            self.h_outputs = self.last_relevant(all_outputs, sequence_length)

        with tf.name_scope("output"):
            l2_loss = tf.constant(0.0)

            W = tf.get_variable("W", shape=[self.cfg.HiddenSize, self.cfg.ClassesCount],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.cfg.ClassesCount]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.output = tf.nn.softmax(logits)
            self.labels = tf.argmax(self.output, axis=1, output_type=dtypes.int32)

        with tf.name_scope("cost"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)

            if self.cfg.UseClassWeights:
                weights = tf.reduce_sum(
                    self.cfg.ClassWeights * tf.one_hot(indices=self.y,
                                                       depth=self.cfg.ClassesCount),
                    axis=1)

                loss = loss * weights

            self.cost = loss + self.cfg.L2RegLambda * l2_loss

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(self.labels, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            Exception("Incorrect cell_type={}".format(cell_type))
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    def create_feed_dict(self, input, data_type):
        return {
            self.x: input[Batch.I_X],
            self.y: input[Batch.I_LABEL],
            self.emb_contexts: input[Batch.I_EMBEDDING],
            self.sentences_inds: input[Batch.I_SENTENCE_INDEX],
            self.dropout_keep_prob: self.cfg.Dropout if data_type == DataType.Train else 1.0
        }

    @property
    def Accuracy(self):
        return self.accuracy

    @property
    def Cost(self):
        return self.cost

    @property
    def Labels(self, debug=True):
        return self.labels

    @property
    def Output(self):
        return self.output

    @property
    def Log(self):

        params = [("dropout", self.dropout_keep_prob),
                  ("labels", self.labels)]

        return [list(p) for p in zip(*params)]

    @property
    def ParametersDictionary(self):
        return {}
