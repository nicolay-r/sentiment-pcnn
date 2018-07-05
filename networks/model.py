import tensorflow as tf
import datetime

import numpy as np
from io import NetworkIO
from callback import Callback

from core.source.opinion import OpinionCollection
from core.evaluation.eval import Evaluator
from core.source.entity import EntityCollection
from core.source.synonyms import SynonymsCollection
from core.runtime.embeddings import Embedding
from core.source.news import News

from networks.architectures.base import NeuralNetwork

from processing.relations import ExtractedRelationsCollection
from processing.indices import EntityIndices
from processing.utils import NewsWords, NewsDescriptor
from processing.batch import BagsCollection
from processing.words import NewsWordsCollection


class TensorflowModel(object, NetworkIO):

    def __init__(self,
                 io,
                 word_embedding,
                 train_indices,
                 test_indices,
                 synonyms_filepath,
                 bag_size,
                 words_per_news,
                 bags_per_minibatch,
                 callback):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(word_embedding, Embedding))
        assert(isinstance(callback, Callback))

        self.io = io
        self.sess = None
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.words_per_news = words_per_news
        self.synonyms_filepath = synonyms_filepath
        self.synonyms = SynonymsCollection.from_file(self.synonyms_filepath)

        # Compute embedding vectors for entities of train and test collections.
        # Initialize entities embedding

        # size of window which includes relations and also filters them.
        # len([ ... entity_1 ... entity_2 ...]) = window_size_in_words
        # TODO. window size should be unchanged

        all_indices = train_indices + test_indices
        entities_collections = [EntityCollection.from_file(self.io.get_entity_filepath(n)) for n in all_indices]
        entity_indices = EntityIndices(entities_collections)

        # Train collection
        train_news_words_collection, train_relations_collection = self._process_into_collections(
            train_indices, entity_indices, word_embedding, words_per_news, True)

        # Test collection
        test_news_words_collection, test_relations_collection = self._process_into_collections(
            test_indices, entity_indices, word_embedding, words_per_news, False)

        words_per_news = min(train_news_words_collection.get_min_words_per_news_count(),
                             test_news_words_collection.get_min_words_per_news_count(),
                             words_per_news)

        self.train_relations_collection = train_relations_collection
        self.test_relations_collection = test_relations_collection
        self.test_news_words_collection = test_news_words_collection

        # Set sample type ...
        sample_type = self.get_sample_type()

        train_bags_collection = BagsCollection(train_relations_collection.relations, bag_size, sample_type=sample_type)
        test_bags_collection = BagsCollection(test_relations_collection.relations, bag_size, sample_type=sample_type)

        train_bags_collection.shuffle()
        test_bags_collection.shuffle()

        self.test_minibatches = test_bags_collection.to_minibatches(bags_per_minibatch)
        self.train_minibatches = train_bags_collection.to_minibatches(bags_per_minibatch)
        self.train_news_words_collection = train_news_words_collection

        self.E = train_news_words_collection.get_embedding_matrix()  # test collection has the same matrix.

        self.network = None
        self.callback = callback

    def set_optimiser(self, optimiser):
        assert(isinstance(self.network, NeuralNetwork))
        self.optimiser = optimiser.minimize(self.network.Cost)

    def get_embedding_shape(self):
        return self.E.shape

    def notify_initialized(self):
        if self.callback is not None:
            self.callback.on_initialized(self)

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.sess.close()

    def initialize_session(self, gpu_memory_fraction=0.25, debug=False):
        """
        Tensorflow session initialization
        """
        if debug:
            print "Initialize tensorflow session ..."
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.sess = sess

    def fit(self, epochs_count, callback=None, debug=False):
        assert(isinstance(epochs_count, int))
        assert(self.sess is not None)

        for e in range(epochs_count):
            total_cost = 0
            np.random.shuffle(self.train_minibatches)
            for it, mbatch in enumerate(self.train_minibatches):
                if debug:
                    print "{}: Epoch {}. Pass minibatch {}/{}...".format(
                        str(datetime.datetime.now()), e + 1, it, len(self.train_minibatches))
                mbatch.shuffle()

                feed_dict = self.create_feed_dict(
                    self.sess,
                    mbatch,
                    self.train_news_words_collection,
                    self.E.shape[0],
                    is_train=True,
                    debug=debug)

                _, cost = self.sess.run([self.optimiser, self.network.Cost],
                                        feed_dict=feed_dict)

                if debug:
                    print "cost type: {}".format(type(cost))
                    print "cost: {}".format(cost)

                total_cost += np.mean(cost)

            if callback is not None:
                callback.on_epoch_finished(total_cost / len(self.train_minibatches))

        if callback is not None:
            callback.on_fit_finished()

    def _process_into_collections(self,
                                  indices,
                                  entity_indices,
                                  word_embedding,
                                  window_size_in_words,
                                  is_train_collection):
        assert(isinstance(indices, list))
        assert(isinstance(word_embedding, Embedding))
        assert(isinstance(is_train_collection, bool))

        rc = ExtractedRelationsCollection()
        nwc = NewsWordsCollection(entity_indices, word_embedding)
        for n in indices:
            assert(type(n) == int)

            entity_filepath = self.io.get_entity_filepath(n)
            news_filepath = self.io.get_news_filepath(n)
            opin_filepath = self.io.get_opinion_input_filepath(n)
            neutral_filepath = self.io.get_neutral_filepath(n, is_train_collection)

            news = News.from_file(news_filepath, EntityCollection.from_file(entity_filepath))

            opinions_collections = [OpinionCollection.from_file(neutral_filepath, self.synonyms_filepath)]
            if is_train_collection:
                opinions_collections.append(OpinionCollection.from_file(opin_filepath, self.synonyms_filepath))

            news_words = NewsWords(n, news)
            news_descriptor = self.create_news_descriptor(n, news, news_words, opinions_collections, is_train_collection)

            rc.add_news_relations(news_descriptor, self.synonyms, window_size_in_words, is_train_collection)
            nwc.add_news(news_words)

        return nwc, rc

    def _apply_model(self, sess, method_name, minibatches, relations_collection,
                     news_words_collection, indices, is_train_collection, debug=False):

        for index, mbatch in enumerate(minibatches):
            feed_dict = self.create_feed_dict(sess, mbatch, news_words_collection, self.E.shape[0], is_train=False, debug=debug)
            uint_labels = sess.run([self.network.Labels], feed_dict=feed_dict)

            if debug:
                print uint_labels

            relations_collection.apply_labels(uint_labels[0].tolist(), mbatch)

        method_root = self.io.get_model_root(method_name)

        collections = relations_collection.to_opinion_collections(indices, self.synonyms)

        for index, opinion_collection in collections:
            assert(isinstance(opinion_collection, OpinionCollection))
            opinion_collection.save(self.io.get_opinion_output_filepath(index, method_root))

        files_to_compare_list = self.io.get_files_to_compare_list(method_name, indices, is_train_collection)

        evaluator = Evaluator(self.synonyms_filepath, method_root)
        return evaluator.evaluate(files_to_compare_list, debug=debug)

    def predict(self, model_name='default_model', test_collection=True, debug=False):

        result = self._apply_model(
            self.sess,
            model_name if test_collection else model_name + '_train',
            self.test_minibatches if test_collection else self.train_minibatches,
            self.test_relations_collection if test_collection else self.train_relations_collection,
            self.test_news_words_collection if test_collection else self.train_news_words_collection,
            self.test_indices if test_collection else self.train_indices,
            debug)

        return result

    def create_feed_dict(self, sess, mbatch, news_words_collection, total_words_count, is_train, debug=False):
        """
        returns: dict
            Returns dictionary for tensorflow session
        """
        raise Exception("Not Implemented")

    def create_model(self):
        """
        returns: processing.nn.NeuralNetwork
            TensorFlow model implementation
        """
        raise Exception("Not Implemented")

    def get_sample_type():
        raise Exception("Not Implemented")

    def create_news_descriptor(self, n, news, news_words, opinions_collections, is_train):
        """
        return: NewsDescriptor
            news descriptor without NLP features vectors
        """
        return NewsDescriptor(n, news, news_words, opinions_collections, [])
