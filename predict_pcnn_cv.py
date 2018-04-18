#!/usr/bin/python
# -*- coding: utf-8 -*- from os.path import join

from os import path
from pcnn import PCNN
from core.runtime.embeddings import RusvectoresEmbedding
from predict_pcnn import PCNNGridCallback
import io_utils


if __name__ == "__main__":

    gpu_memory_fraction=0.25
    # where to save results

    # Fixed values
    epochs = 251
    test_on_epochs=range(0, 251, 25)
    cv_count = 3
    adadelta_epsilon = 10e-6            # according to paper (Zeiler, 2012)
    adadelta_rho = 0.95                 # according to paper (Zeiler, 2012)
    dropout = 0.5
    words_per_news = 15                 # because of a time this is a predefined parameter
    bags_per_minibatch = 50
    position_size = 5                   # little affect on result, heuristically choose
    filters = [200]
    csv_filepath = path.join(io_utils.data_root(), 'pcnn_grid.csv')
    embedding = RusvectoresEmbedding(io_utils.load_w2v_model())

    callback = PCNNGridCallback(epochs=epochs,
                                test_on_epochs=test_on_epochs,
                                csv_filepath=csv_filepath,
                                model_name='pcnn_grid_cv')

    # Grid parameters
    windows = [2, 3, 5, 7, 10]

    for w in windows:
        for f in filters:
            for _, pair in enumerate(io_utils.indices_to_cv_pairs(cv_count)):
                train_indices, test_indices = pair

                model = PCNN(embedding=embedding,
                             words_per_news=words_per_news,
                             train_indices=train_indices,
                             test_indices=test_indices,
                             position_size=position_size,
                             window_size=w,
                             channels_count=f,
                             bags_per_minibatch=bags_per_minibatch,
                             adadelta_rho=adadelta_rho,
                             adadelta_epsilon=adadelta_epsilon,
                             dropout=dropout,
                             callback=callback)

                model.initialize_session(gpu_memory_fraction=gpu_memory_fraction)
                model.fit(epochs, callback=callback)
                model.dispose_session()
