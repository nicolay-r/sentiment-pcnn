#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
from os import path

from model_cnn import CNNModel
from model_pcnn import PCNNModel
from pandas_logger_callback import PandasLoggerCallback

from networks.callback import Callback
from networks.configurations.cnn import CNNConfig

from core.runtime.embeddings import RusvectoresEmbedding

import io_utils


if __name__ == "__main__":

    gpu_memory_fraction = 0.15

    cnn_filepath = path.join(io_utils.data_root(), 'cnn_grid.csv')
    pcnn_filepath = path.join(io_utils.data_root(), 'pcnn_grid.csv')

    config = CNNConfig()

    cnn_callback = PandasLoggerCallback(epochs=config.Epochs,
                                        test_on_epochs=config.test_on_epochs,
                                        csv_filepath=cnn_filepath,
                                        model_name='cnn_grid')
    pcnn_callback = PandasLoggerCallback(epochs=config.Epochs,
                                         test_on_epochs=config.test_on_epochs,
                                         csv_filepath=pcnn_filepath,
                                         model_name='pcnn_grid')

    embedding = RusvectoresEmbedding(io_utils.load_w2v_model())
    train_indices = io_utils.train_indices()
    test_indices = io_utils.test_indices()

    models = [
        (CNNModel, cnn_callback),
        (PCNNModel, pcnn_callback)]

    for _ in config.iterate_over_grid():
        for tf_model, callback in models:
            assert(isinstance(callback, Callback))

            model = tf_model(word_embedding=embedding,
                             train_indices=train_indices,
                             test_indices=test_indices,
                             bag_size=config.bag_size,
                             words_per_news=config.words_per_news,
                             bags_per_minibatch=config.bags_per_minibatch,
                             callback=callback)

            model.compile_network(config)
            model.set_optimiser(config.optimiser)
            model.notify_initialized()

            model.initialize_session(gpu_memory_fraction=gpu_memory_fraction)
            model.fit(config.Epochs, callback=callback)
            model.dispose_session()
