#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path

from model_cnn import CNNModel
from model_pcnn import PCNNModel
from pandas_logger_callback import PandasLoggerCallback

from networks.callback import Callback
from networks.configurations.cnn import CNNConfig

import io_utils


def eval_model(model_type, config, callback,
               train_indices, test_indices,
               gpu_memory_fraction=0.15):

    assert(isinstance(callback, Callback))
    assert(isinstance(config, CNNConfig))
    assert(isinstance(train_indices, list))
    assert(isinstance(test_indices, list))

    config.use_nlp_vector = False

    model = model_type(word_embedding=config.embedding,
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


def compose_all_models(config,
                       output_root=io_utils.data_root(),
                       file_suffix_name=''):

    assert(isinstance(config, CNNConfig))
    assert(isinstance(file_suffix_name, str))

    cnn_name = 'cnn_grid{}'.format(file_suffix_name)
    pcnn_name = 'pcnn_grid{}'.format(file_suffix_name)

    cnn_filepath = path.join(output_root, '{}.csv'.format(cnn_name))
    pcnn_filepath = path.join(output_root, '{}.csv'.format(pcnn_name))

    cnn_callback = PandasLoggerCallback(epochs=config.Epochs,
                                        test_on_epochs=config.test_on_epochs,
                                        csv_filepath=cnn_filepath,
                                        model_name=cnn_name)
    pcnn_callback = PandasLoggerCallback(epochs=config.Epochs,
                                         test_on_epochs=config.test_on_epochs,
                                         csv_filepath=pcnn_filepath,
                                         model_name=pcnn_name)

    return [(CNNModel, cnn_callback),
            (PCNNModel, pcnn_callback)]


if __name__ == "__main__":

    gpu_memory_fraction = 0.35

    config = CNNConfig()
    models = compose_all_models(config)

    for _ in config.iterate_over_grid():
        for tf_model, callback in models:
            eval_model(config=config,
                       callback=callback,
                       model_type=tf_model,
                       train_indices=io_utils.train_indices(),
                       test_indices=io_utils.test_indices(),
                       gpu_memory_fraction=gpu_memory_fraction)
