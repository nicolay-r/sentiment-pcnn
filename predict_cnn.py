#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import tensorflow as tf
from os import path

from model_cnn import CNN
from model_pcnn import PCNN

from networks.architectures.cnn import VanillaCNN
from networks.architectures.pcnn import PiecewiseCNN
from networks.callback import Callback
from networks.logger import PandasResultLogger

from core.runtime.embeddings import RusvectoresEmbedding

import io_utils


class PandasLoggerCallback(Callback):

    col_train_relations_count = 'train_relations'
    col_test_relations_count = 'test_relations'
    col_epochs = 'epochs'

    def __init__(self,
                 epochs,
                 test_on_epochs,
                 csv_filepath,
                 model_name):
        """
        epochs: int
           amount of epochs to train
        test_on_epochs: list
            list of epochs at which it is necessary to call test.
        csv_filepath: str
            output filepath
        model_name: str
        """
        self.logger = PandasResultLogger(test_on_epochs)
        self.logger.add_column_if_not_exists(self.col_train_relations_count, int)
        self.logger.add_column_if_not_exists(self.col_test_relations_count, int)
        self.logger.add_column_if_not_exists(self.col_epochs, int)

        self.model = None
        self.current_epoch = 0
        self.epochs = epochs
        self.test_on_epochs = test_on_epochs
        self.csv_filepath = csv_filepath
        self.model_name = model_name

    def on_initialized(self, model):

        self.current_epoch = 0
        row = self.logger.create_new_row()

        row[self.col_train_relations_count] = len(model.train_relations_collection.relations)
        row[self.col_test_relations_count] = len(model.test_relations_collection.relations)
        row[self.col_epochs] = self.epochs

        for key, value in model.network.ParametersDictionary.iteritems():
            col_name = "col_{}".format(key)
            self.logger.add_column_if_not_exists(col_name, type(value))
            self.logger.write_value(col_name, value, debug=True)

        self.model = model

    def on_epoch_finished(self, avg_cost):

        print "avg_cost type: {}".format(type(avg_cost))
        print "avg_cost value: '{}'".format(avg_cost)

        self.current_epoch += 1
        print "{}: Epoch: {}, average cost: {:.3f}".format(
            str(datetime.datetime.now()), self.current_epoch, avg_cost)

        if self.current_epoch not in self.test_on_epochs:
            return

        self.logger.write_evaluation_results(
            current_epoch=self.current_epoch,
            result_test=self.model.predict(model_name=self.model_name, test_collection=True, debug=True),
            result_train=self.model.predict(model_name=self.model_name, test_collection=False, debug=True),
            avg_cost=avg_cost)

        # save
        self.logger.df.to_csv(self.csv_filepath)


if __name__ == "__main__":

    gpu_memory_fraction = 0.15

    # Fixed values
    epochs = 251
    test_on_epochs = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    window_size = 3
    bag_size = 1
    position_size = 1
    bags_per_minibatch = 50
    dropout = 0.5

    learning_rate = 0.1
    adadelta_epsilon = 10e-6    # according to paper (Zeiler, 2012)
    adadelta_rho = 0.95         # according to paper (Zeiler, 2012)

    optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=learning_rate,
        epsilon=adadelta_epsilon,
        rho=adadelta_rho)

    cnn_filepath = path.join(io_utils.data_root(), 'cnn_grid.csv')
    pcnn_filepath = path.join(io_utils.data_root(), 'pcnn_grid.csv')

    cnn_callback = PandasLoggerCallback(epochs=epochs, test_on_epochs=test_on_epochs, csv_filepath=cnn_filepath, model_name='cnn_grid')
    pcnn_callback = PandasLoggerCallback(epochs=epochs, test_on_epochs=test_on_epochs, csv_filepath=pcnn_filepath, model_name='pcnn_grid')

    embedding = RusvectoresEmbedding(io_utils.load_w2v_model())

    # Grid values (significant parameters)
    words_per_news = [50, 100, 150, 200]
    filters_set = [100, 200, 300]
    architectures = [(CNN, cnn_callback), (PCNN, pcnn_callback)]

    for wpn in words_per_news:
        for channels_count in filters_set:
            for tf_model, callback in architectures:
                assert(isinstance(callback, Callback))

                model = tf_model(word_embedding=embedding,
                                 train_indices=io_utils.train_indices(),
                                 test_indices=io_utils.test_indices(),
                                 bag_size=bag_size,
                                 words_per_news=wpn,
                                 bags_per_minibatch=bags_per_minibatch,
                                 callback=callback)

                embedding_shape = model.get_embedding_shape()

                compiled_network = None
                if isinstance(model, PCNN):
                    compiled_network = PiecewiseCNN(
                        vocabulary_words=embedding_shape[0],
                        embedding_size=embedding_shape[1],
                        words_per_news=model.words_per_news,
                        bags_per_batch=bags_per_minibatch,
                        bag_size=bag_size,
                        channels_count=channels_count,
                        window_size=window_size,
                        dp=position_size,
                        dropout=dropout)
                elif isinstance(model, CNN):
                    compiled_network = VanillaCNN(
                        vocabulary_words=embedding_shape[0],
                        embedding_size=embedding_shape[1],
                        words_per_news=model.words_per_news,
                        bags_per_batch=bags_per_minibatch,
                        bag_size=bag_size,
                        channels_count=channels_count,
                        window_size=window_size,
                        dp=position_size,
                        dropout=dropout)

                print type(compiled_network)
                model.set_compiled_network(compiled_network)
                model.set_optimiser(optimiser)
                model.notify_initialized()

                model.initialize_session(gpu_memory_fraction=gpu_memory_fraction, debug=True)
                model.fit(epochs, callback=callback, debug=True)
                model.dispose_session()
