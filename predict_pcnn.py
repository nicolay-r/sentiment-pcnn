#!/usr/bin/python
# -*- coding: utf-8 -*- from os.path import join

import pandas as pd
import datetime
from os import path

from pcnn import PCNN
from networks.callback import Callback
from core.runtime.embeddings import RusvectoresEmbedding
from core.evaluation.eval import Evaluator
import io_utils


class PCNNGridCallback(Callback):

    col_embedding_size = 'embedding_size'
    col_word_per_news = 'words_per_news'
    col_window_size = 'window_size'
    col_bag_size = 'bag_size'
    col_bags_per_minibatch = 'bags_per_minibatch'
    col_filters_count = 'filters_count'
    col_train_relations_count = 'train_relations'
    col_test_relations_count = 'test_relations'
    col_position_size = 'position_size'
    col_dropout = 'dropout'
    col_epochs = 'epochs'

    col_f1 = 'f1_{}'
    col_f1p = 'f1-p_{}'
    col_f1n = 'f1-n_{}'
    col_avg_cost = 'avg-cost_{}'
    col_pp = 'pos-p_{}'
    col_np = 'neg-p_{}'
    col_pr = 'pos-r_{}'
    col_nr = 'neg-r_{}'
    col_f1_train = 'f1-train_{}'

    def __init__(self,
                 epochs,
                 test_on_epochs,
                 csv_filepath,
                 model_name):

        columns = [self.col_embedding_size,
                   self.col_word_per_news,
                   self.col_window_size,
                   self.col_bag_size,
                   self.col_bags_per_minibatch,
                   self.col_filters_count,
                   self.col_train_relations_count,
                   self.col_test_relations_count,
                   self.col_position_size,
                   self.col_dropout,
                   self.col_epochs]

        for i in test_on_epochs:
            columns.append(self._get_f1_column_name(i))
            columns.append(self._get_f1p_column_name(i))
            columns.append(self._get_f1n_column_name(i))
            columns.append(self._get_pp_column_name(i))
            columns.append(self._get_np_column_name(i))
            columns.append(self._get_pr_column_name(i))
            columns.append(self._get_nr_column_name(i))
            columns.append(self._get_f1_train_column_name(i))
            columns.append(self._get_avg_cost_column_name(i))

        self.df = pd.DataFrame(columns=columns)

        self.test_on_epochs = test_on_epochs
        self.epoch_index = 0
        self.model_index = 0
        self.epochs = epochs
        self.csv_filepath = csv_filepath
        self.model_name = model_name

    @staticmethod
    def _get_f1_column_name(index):
        return PCNNGridCallback.col_f1.format(index)

    @staticmethod
    def _get_avg_cost_column_name(index):
        return PCNNGridCallback.col_avg_cost.format(index)

    @staticmethod
    def _get_f1p_column_name(index):
        return PCNNGridCallback.col_f1p.format(index)

    @staticmethod
    def _get_f1n_column_name(index):
        return PCNNGridCallback.col_f1n.format(index)

    @staticmethod
    def _get_pp_column_name(index):
        return PCNNGridCallback.col_pp.format(index)

    @staticmethod
    def _get_np_column_name(index):
        return PCNNGridCallback.col_np.format(index)

    @staticmethod
    def _get_pr_column_name(index):
        return PCNNGridCallback.col_pr.format(index)

    @staticmethod
    def _get_nr_column_name(index):
        return PCNNGridCallback.col_nr.format(index)

    @staticmethod
    def _get_f1_train_column_name(index):
        return PCNNGridCallback.col_f1_train.format(index)

    def on_initialized(self, pcnn):
        assert(isinstance(pcnn, PCNN))

        self.e = 0
        self.model_index += 1

        self.df.loc[self.model_index] = None
        row = self.df.loc[self.model_index]

        row[self.col_embedding_size] = pcnn.E.shape[1]
        row[self.col_word_per_news] = pcnn.words_per_news
        row[self.col_window_size] = pcnn.window_size
        row[self.col_position_size] = pcnn.position_size
        row[self.col_bag_size] = pcnn.bag_size
        row[self.col_bags_per_minibatch] = pcnn.bags_per_minibatch
        row[self.col_filters_count] = pcnn.channels_count
        row[self.col_dropout] = pcnn.dropout
        row[self.col_train_relations_count] = len(pcnn.train_relations_collection.relations)
        row[self.col_test_relations_count] = len(pcnn.test_relations_collection.relations)
        row[self.col_epochs] = self.epochs

        self.pcnn = pcnn

    def on_epoch_finished(self, avg_cost):
        self.e += 1

        print "{}: Epoch: {}, average cost: {:.3f}".format(
            str(datetime.datetime.now()), self.e, avg_cost)

        if self.e not in self.test_on_epochs:
            return

        i = self.e
        result_test = self.pcnn.predict(model_name=self.model_name, test_collection=True)
        result_train = self.pcnn.predict(model_name=self.model_name, test_collection=False)

        # apply results
        self.df.loc[self.model_index][self._get_f1_column_name(i)] = result_test[Evaluator.C_F1]
        self.df.loc[self.model_index][self._get_f1p_column_name(i)] = result_test[Evaluator.C_F1_POS]
        self.df.loc[self.model_index][self._get_f1n_column_name(i)] = result_test[Evaluator.C_F1_NEG]
        self.df.loc[self.model_index][self._get_pp_column_name(i)] = result_test[Evaluator.C_POS_PREC]
        self.df.loc[self.model_index][self._get_np_column_name(i)] = result_test[Evaluator.C_NEG_PREC]
        self.df.loc[self.model_index][self._get_pr_column_name(i)] = result_test[Evaluator.C_POS_RECALL]
        self.df.loc[self.model_index][self._get_nr_column_name(i)] = result_test[Evaluator.C_NEG_RECALL]
        self.df.loc[self.model_index][self._get_f1_train_column_name(i)] = result_train[Evaluator.C_F1]

        self.df.loc[self.model_index][self._get_avg_cost_column_name(i)] = avg_cost
        self.df.to_csv(self.csv_filepath)


if __name__ == "__main__":

    gpu_memory_fraction=0.15

    # Fixed values
    epochs = 251
    test_on_epochs=range(0, 251, 25)
    # test_on_epochs=[1,2]
    window_size = 3
    bag_size = 1
    position_size = 1
    adadelta_epsilon = 10e-6            # according to paper (Zeiler, 2012)
    adadelta_rho = 0.95                 # according to paper (Zeiler, 2012)
    bags_per_minibatch = 50
    dropout = 0.5
    csv_filepath = path.join(io_utils.data_root(), 'collection_pcnn_grid.csv')

    callback = PCNNGridCallback(epochs=epochs,
                                test_on_epochs=test_on_epochs,
                                csv_filepath=csv_filepath,
                                model_name='collection_pcnn_grid')

    embedding = RusvectoresEmbedding(io_utils.load_w2v_model())

    # Grid values (significant parameters)
    words_per_news = [25, 50, 100, 150, 200]
    filters_set = [100, 150, 200, 250]

    for wpn in words_per_news:
        for f in filters_set:
            model = PCNN(embedding=embedding,
                         train_indices=io_utils.train_indices(),
                         test_indices=io_utils.test_indices(),
                         bag_size=bag_size,
                         words_per_news=wpn,
                         window_size=window_size,
                         channels_count=f,
                         bags_per_minibatch=bags_per_minibatch,
		    	         position_size=position_size,
                         adadelta_rho=adadelta_rho,
                         adadelta_epsilon=adadelta_epsilon,
                         dropout=dropout,
                         callback=callback)

            model.initialize_session(gpu_memory_fraction=gpu_memory_fraction)
            model.fit(epochs, callback=callback)
            model.dispose_session()
