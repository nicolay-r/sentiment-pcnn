#!/usr/bin/python
import tensorflow as tf
from tensorflow.python.framework.tensor_util import constant_value_as_shape


def _splitting(i, p1_ind, p2_ind, bwc_conv, channels_count, outputs):
    l_ind = tf.minimum(tf.gather(p1_ind, [i]), tf.gather(p2_ind, [i]))  # left
    r_ind = tf.maximum(tf.gather(p1_ind, [i]), tf.gather(p2_ind, [i]))  # right

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
    return i, p1_ind, p2_ind, bwc_conv, channels_count, outputs


def create_pcnn(
    vocabulary_words,       # vocabulary size
    embedding_size,         # word's vector dimention
    words_per_news,         # max amount of words per news
    bags_per_batch=1,
    bag_size=5,
    channels_count=200,     # amount of channels of conv kernels
    window_size=7,          # conv kernel parameter
    dp=5,                   # dimension of the position
    n_out=3,
    dropout=0.5):
    """
    returns: tuple of tf.Tensor
        input_params, cost, labels

    Note: returned 'labels' has a CONVERTED LABELS, where values as follows:
        0 -- negative, 1 -- neutral, and 2 -- positive.
    """
    batch_size = bag_size * bags_per_batch
    dp = 1
    embedding_size_p = embedding_size + 2 * dp  # embedding + dp_1 + dp_2

    # Hidden state variables
    W = tf.Variable(tf.random_normal([3 * channels_count, n_out]), dtype=tf.float32)
    bias = tf.Variable(tf.random_normal([n_out]), dtype=tf.float32)
    conv_filter = tf.Variable(tf.random_normal([window_size * embedding_size_p, 1, channels_count]), dtype=tf.float32)
    E_d = tf.Variable(tf.random_normal([vocabulary_words, dp]))

    # Input placeholders
    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, words_per_news])
    P1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, words_per_news])
    P2 = tf.placeholder(dtype=tf.float32, shape=[batch_size, words_per_news])
    p1_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size])  # left indices for each batch
    p2_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size])  # right indices for each batch
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    E = tf.placeholder(dtype=tf.float32, shape=[vocabulary_words, embedding_size])
    # bernoulli = tf.distributions.Bernoulli(dropout, dtype=tf.float32)

    # constants
    # p_P1 = tf.nn.embedding_lookup(E_d, P1)
    p_P1 = tf.reshape(P1, [batch_size, words_per_news, 1])
    p_P1 = tf.pad(p_P1, [[0, 0], [0, 0], [embedding_size, dp]])

    # p_P2 = tf.nn.embedding_lookup(E_d, P2)
    p_P2 = tf.reshape(P2, [batch_size, words_per_news, 1])
    p_P2 = tf.pad(p_P2, [[0, 0], [0, 0], [embedding_size+dp, 0]])

    # apply embeding for input x indices
    e = tf.nn.embedding_lookup(E, x)

    e_p1p2 = tf.pad(e, [[0, 0], [0, 0], [0, 2*dp]])
    e_p1p2 = tf.add(e_p1p2, p_P1)
    e_p1p2 = tf.add(e_p1p2, p_P2)

    # add padding embedding rows (reason -- have the same amount of rows after conv1d).
    left_padding = (window_size - 1) / 2
    right_padding = (window_size - 1) - left_padding
    e_p1p2 = tf.pad(e_p1p2, [[0, 0], [left_padding, right_padding], [0, 0]])

    # concatenate rows of matrix
    bwc_line = tf.reshape(e_p1p2, [batch_size, (words_per_news + (window_size - 1)) * embedding_size_p, 1])
    bwc_conv = tf.nn.conv1d(bwc_line, conv_filter, embedding_size_p, "VALID", data_format="NHWC", name="conv")

    # slice all data into 3 parts -- before, inner, and after according to relation
    sliced = tf.TensorArray(dtype=tf.float32, size=batch_size, infer_shape=False, dynamic_size=True)
    _, _, _, _, _, sliced = tf.while_loop(
            lambda i, *_: tf.less(i, batch_size),
            _splitting,
            [0, p1_ind, p2_ind, bwc_conv, channels_count, sliced])
    sliced = tf.squeeze(sliced.concat())

    # maxpool
    bwgc_mpool = tf.nn.max_pool(
            sliced,
            [1, 1, words_per_news, 1],
            [1, 1, words_per_news, 1],
            padding='VALID',
            data_format="NHWC")

    bwc_mpool = tf.squeeze(bwgc_mpool, [2])

    bcw_mpool = tf.transpose(bwc_mpool, perm=[0, 2, 1])
    bc_pmpool = tf.reshape(bcw_mpool, [batch_size, 3*channels_count])
    g = tf.tanh(bc_pmpool)

    # apply Bernoulli mask for 'g'
    # r = bernoulli.sample(sample_shape=[1, 3*channels_count])
    # r_batch = tf.matmul(tf.constant(1, shape=[batch_size, 1], dtype=tf.float32), r)
    # masked_g = tf.multiply(g, r_batch)

    logits_unscaled = tf.matmul(g, W) + bias
    logits_unscaled_dropout = tf.nn.dropout(logits_unscaled, dropout)

    labels = tf.argmax(tf.nn.softmax(logits_unscaled), axis=1)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_unscaled_dropout,
            labels=y)
    cross_entropy_per_bag = tf.reshape(cross_entropy, [bags_per_batch, bag_size])

    cost = tf.reduce_max(cross_entropy_per_bag, axis=1)
    return x, P1, P2, p1_ind, p2_ind, y, E, cost, labels
