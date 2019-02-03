import tensorflow as tf
import numpy as np

from networks.context.configurations.base import CommonModelSettings


def merge_with_embedding(embedding, vectors):
    """
    Merges embedding layer with additional features, where each feature
    is a vector of an embedding_size.

    embedding: tf object
        words embedding of shape [batch_size, sequence_length, embedding_size]

    vectors: list
        list of vectors, where shape of each vector is [batch_size, sequence_length]
    """
    assert(isinstance(vectors, list))

    for index, v in enumerate(vectors):
        vectors[index] = tf.expand_dims(v, -1)

    return tf.concat([embedding] + vectors, axis=-1)


def to_one_hot(y_list, size):
    y = np.array(y_list)
    y_one_hot = np.zeros((y.size, size))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def get_two_layer_logits(g, W, bias, W2, bias2):
    r = tf.matmul(g, W) + bias
    return tf.matmul(r, W2) + bias2


def init_weighted_cost(logits_unscaled_dropout, true_labels, cfg):
    """
    Init loss with weights for tensorflow model.
    'labels' suppose to be a list of indices (not priorities)
    """
    assert(isinstance(cfg, CommonModelSettings))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_unscaled_dropout,
        labels=true_labels)

    weights = tf.reduce_sum(
        cfg.ClassWeights * tf.one_hot(indices=true_labels, depth=cfg.ClassesCount),
        axis=1)

    if cfg.UseClassWeights:
        loss = loss * weights

    loss = tf.reshape(loss, [cfg.BagsPerMinibatch, cfg.BagSize])
    cost = tf.reduce_max(loss, axis=1)

    return weights, cost

def init_embedded_terms(x,
                        dist_from_subj,
                        dist_from_obj,
                        pos,
                        cfg):
    """
    Compose result embedding based on:
    'x', 'dist_from_subj', 'dist_from_obj', and 'pos'
    """
    assert(isinstance(cfg, CommonModelSettings))

    term_emb = tf.constant(value=cfg.TermEmbeddingMatrix,
                           dtype=tf.float32,
                           shape=cfg.TermEmbeddingShape)

    dist_emb = tf.get_variable(dtype=tf.float32,
                               initializer=tf.random_normal_initializer,
                               shape=[cfg.TermsPerContext, cfg.DistanceEmbeddingSize],
                               trainable=True,
                               name="dist_emb")

    pos_emb = tf.get_variable(dtype=tf.float32,
                              initializer=tf.random_normal_initializer,
                              shape=[len(cfg.PosTagger.pos_names), cfg.PosEmbeddingSize],
                              trainable=True,
                              name="pos_emb")

    embedded_terms = tf.concat([tf.nn.embedding_lookup(term_emb, x),
                                tf.nn.embedding_lookup(dist_emb, dist_from_subj),
                                tf.nn.embedding_lookup(dist_emb, dist_from_obj)],
                               axis=-1)

    if cfg.UsePOSEmbedding:
        embedded_terms = tf.concat([embedded_terms,
                                    tf.nn.embedding_lookup(pos_emb, pos)],
                                   axis=-1)

    return embedded_terms


def init_accuracy(labels, true_labels):
    correct = tf.equal(labels, true_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))



