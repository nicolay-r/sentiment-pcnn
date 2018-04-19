import numpy as np
import tensorflow as tf

from core.evaluation.eval import Evaluator
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection
from core.source.synonyms import SynonymsCollection
from core.runtime.embeddings import Embedding

from networks.model import Model
from networks.pcnn import pcnn_core
from networks.callback import Callback

from networks.core.relations import ExtractedRelationCollection
from networks.core.indices import EntityIndices
from networks.core.utils import NewsDescriptor, NewsWords
from networks.core.batch import BagsCollection
from networks.core.words import NewsWordsCollection

import io_utils


class PCNN(Model):

    def __init__(
            self,
            embedding,
            synonyms_filepath=io_utils.get_synonyms_filepath(),
            train_indices=io_utils.train_indices(),
            test_indices=io_utils.test_indices(),
            words_per_news=25,
            window_size=3,
            bag_size=1,
            position_size=7,
            bags_per_minibatch=50,
            learning_rate=0.1,
            channels_count=25,
            adadelta_epsilon=10e-6,
            adadelta_rho=0.95,
            dropout=0.5,
            callback=None):

        assert(isinstance(embedding, Embedding))
        assert(isinstance(callback, Callback))

        self.sess = None
        self.channels_count = channels_count
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.words_per_news = words_per_news
        self.window_size = window_size
        self.position_size = position_size
        self.synonyms_filepath = synonyms_filepath
        self.synonyms = SynonymsCollection.from_file(self.synonyms_filepath)
        self.bag_size = bag_size
        self.bags_per_minibatch = bags_per_minibatch
        self.dropout = dropout

        # Compute embedding vectors for entities of train and test collections.
        # Initialize entities embedding
        all_indices = train_indices + test_indices
        entities_collections = [EntityCollection.from_file(io_utils.get_entity_filepath(n))
                                for n in all_indices]
        entities_embedding = EntityIndices(entities_collections)

        # size of window which includes relations and also filters them.
        # len([ ... entity_1 ... entity_2 ...]) = window_size_in_words
        # TODO. window size should be unchanged

        # Train collection
        train_news_words_collection, train_relations_collection = self._process_into_collections(
            train_indices, entities_embedding, embedding, words_per_news, True)
        # Test collection
        test_news_words_collection, test_relations_collection = self._process_into_collections(
            test_indices, entities_embedding, embedding, words_per_news, False)

        words_per_news = min(train_news_words_collection.get_min_words_per_news_count(),
                             test_news_words_collection.get_min_words_per_news_count(),
                             words_per_news)

        # Create bags ...
        train_bags_collection = BagsCollection(train_relations_collection.relations, bag_size)
        test_bags_collection = BagsCollection(test_relations_collection.relations, bag_size)

        train_bags_collection.shuffle()
        test_bags_collection.shuffle()

        # Split into train_minibatches
        self.train_minibatches = train_bags_collection.to_minibatches(bags_per_minibatch)
        self.test_minibatches = test_bags_collection.to_minibatches(bags_per_minibatch)

        self.train_relations_collection = train_relations_collection
        self.test_relations_collection = test_relations_collection
        self.train_news_words_collection = train_news_words_collection
        self.test_news_words_collection = test_news_words_collection

        # Prepare input parameters
        E = train_news_words_collection.get_embedding_matrix()  # test collection has the same matrix.

        self.E = E

        self.in_x, self.in_P1, self.in_P2, self.in_p1_ind, self.in_p2_ind, self.in_y, self.in_E, self.cost, self.labels = pcnn_core.create_pcnn(
            vocabulary_words=E.shape[0],
            embedding_size=E.shape[1],
            words_per_news=words_per_news,
            bags_per_batch=bags_per_minibatch,
            bag_size=bag_size,
            channels_count=channels_count,
            window_size=window_size,
            dp=position_size,
            dropout=dropout)

        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                    epsilon=adadelta_epsilon,
                                                    rho=adadelta_rho).minimize(self.cost)

        # Initialization finished
        if callback is not None:
            callback.on_initialized(self)

    def _process_into_collections(self,
                                  indices,
                                  entities_embedding,
                                  embedding,
                                  window_size_in_words,
                                  is_train_collection):
        assert(type(indices) == list)
        assert(isinstance(embedding, Embedding))
        assert(type(is_train_collection) == bool)

        rc = ExtractedRelationCollection()
        nwc = NewsWordsCollection(entities_embedding, embedding)
        for n in indices:
            assert(type(n) == int)

            entity_filepath = io_utils.get_entity_filepath(n)
            news_filepath = io_utils.get_news_filepath(n)
            opin_filepath = io_utils.get_opin_filepath(n, is_etalon=True)
            neutral_filepath = io_utils.get_neutral_filepath(n, is_train=is_train_collection)

            news = News.from_file(news_filepath, EntityCollection.from_file(entity_filepath))

            opinions_collections = [OpinionCollection.from_file(neutral_filepath, self.synonyms_filepath)]
            if is_train_collection:
                opinions_collections.append(OpinionCollection.from_file(opin_filepath, self.synonyms_filepath))

            news_words = NewsWords(n, news)
            news_descriptor = NewsDescriptor(n, news, news_words, opinions_collections)

            rc.add_news_relations(news_descriptor, self.synonyms, window_size_in_words, is_train_collection)
            nwc.add_news(news_words)

        return nwc, rc

    def _apply_pcnn_model(self, sess, method_name, minibatches, relations_collection,
                          news_words_collection, indices, is_train_collection):

        total_words_count = self.E.shape[0]
        for index, mbatch in enumerate(minibatches):
            X, p1, p2, P1, P2, y = mbatch.to_network_input(
                news_words_collection, total_words_count, self.words_per_news)
            uint_labels = sess.run([self.labels],
                                   feed_dict={
                                       self.in_x: X,
                                       self.in_y: y,
                                       self.in_P1: P1,
                                       self.in_P2: P2,
                                       self.in_p1_ind: p1,
                                       self.in_p2_ind: p2,
                                       self.in_E: self.E})
            relations_collection.apply_labels(uint_labels[0].tolist(), mbatch)

        method_root = io_utils.get_method_root(method_name)

        # print "Getting opinion collections ..."
        collections = relations_collection.to_opinion_collections(indices, self.synonyms)

        # print "Save ..."
        for index, opinion_collection in collections:
            assert(isinstance(opinion_collection, OpinionCollection))
            opin_filepath = io_utils.get_opin_filepath(index, is_etalon=False, root=method_root)
            opinion_collection.save(opin_filepath)

        # print "Evaluate ..."
        files_to_compare_list = io_utils.create_files_to_compare_list(
            method_name, indices=indices, is_train_collection=is_train_collection)

        evaluator = Evaluator(self.synonyms_filepath, method_root)
        r = evaluator.evaluate(files_to_compare_list)
        return r

    def initialize_session(self, gpu_memory_fraction=0.25):
        print "Initialize tensorflow session ..."
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.sess = sess

    def dispose_session(self):
        self.sess.close()

    def fit(self, epochs, callback=None):
        assert(type(epochs) == int)
        assert(self.sess is not None)

        total_words_count = self.E.shape[0]
        for e in range(epochs):
            avg_cost = 0
            np.random.shuffle(self.train_minibatches)
            for it, mbatch in enumerate(self.train_minibatches):
                # print "{}: Epoch {}. Pass minibatch {}/{}...".format(
                #     str(datetime.datetime.now()), e + 1, it, len(train_minibatches))
                mbatch.shuffle()
                X, p1, p2, P1, P2, y = mbatch.to_network_input(
                    self.train_news_words_collection,
                    total_words_count,
                    self.words_per_news)

                _, cost_list = self.sess.run([self.optimiser, self.cost],
                                             feed_dict={
                                                 self.in_x: X,
                                                 self.in_y: y,
                                                 self.in_P1: P1,
                                                 self.in_P2: P2,
                                                 self.in_p1_ind: p1,
                                                 self.in_p2_ind: p2,
                                                 self.in_E: self.E})
                # print cost_list
                avg_cost += (sum(cost_list) / len(cost_list)) / len(self.train_minibatches)

            if callback is not None:
                callback.on_epoch_finished(avg_cost)
        if callback is not None:
            callback.on_fit_finished()

    def predict(self, model_name='pcnn', test_collection=True):

        result = self._apply_pcnn_model(
            self.sess,
            model_name if test_collection else model_name + '_train',
            self.test_minibatches if test_collection else self.train_minibatches,
            self.test_relations_collection if test_collection else self.train_relations_collection,
            self.test_news_words_collection if test_collection else self.train_news_words_collection,
            self.test_indices if test_collection else self.train_indices,
            False)

        return result
