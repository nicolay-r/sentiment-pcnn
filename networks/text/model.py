import numpy as np

from io import TextLevelNetworkIO
from debug import TextDebugKeys

from core.evaluation.labels import Label
from core.source.synonyms import SynonymsCollection
from core.source.opinion import OpinionCollection
from core.source.news import News

from networks.model import TensorflowModel
from networks.io import DataType
from networks.context.processing.extracted_relations import ExtractedRelationsCollection, ExtractedRelation
from networks.context.processing.predictions import RelationPredictionResultCollection

from networks.text.configurations.base import TextModelSettings
from networks.context.configurations.base import CommonModelSettings

from processing.batch import Batch

import io_utils


class TextLevelTensorflowModel(TensorflowModel):

    def __init__(self, io, settings, callback):
        assert(isinstance(io, TextLevelNetworkIO))
        assert(isinstance(settings, TextModelSettings))
        super(TextLevelTensorflowModel, self).__init__(io, callback)

        self._settings = settings
        self.synonyms = SynonymsCollection.from_file(io.get_synonyms_collection_filepath(),
                                                     stemmer=self.Settings.Stemmer)

        contextSettings = CommonModelSettings(load_embedding=False)

        self.relation_collections = {
            DataType.Train: ExtractedRelationsCollection.load(io.get_relations_filepath(self._settings.EpochToUse, DataType.Train)),
            DataType.Test: ExtractedRelationsCollection.load(io.get_relations_filepath(self._settings.EpochToUse, DataType.Test))
        }

        self.predicted_collections = {
            DataType.Train: RelationPredictionResultCollection.load(
                io.get_relations_predictions_filepath(self._settings.EpochToUse, DataType.Train)),
            DataType.Test: RelationPredictionResultCollection.load(
                io.get_relations_predictions_filepath(self._settings.EpochToUse, DataType.Test))
        }

        assert(len(self.predicted_collections[DataType.Train]) == len(self.relation_collections[DataType.Train]))
        assert(len(self.predicted_collections[DataType.Test]) == len(self.relation_collections[DataType.Test]))

        settings.set_embedding_shape((settings.GroupSize * settings.BatchSize + 1, settings.ClassesCount))

        neu, pos, neg = self.relation_collections[DataType.Train].get_statistic()
        settings.set_class_weights([100.0 / (neu + 1), 100.0 / (pos + 1), 100.0 / (neg + 1)])

        # log
        keys, values = settings.get_parameters()
        self._display_log(keys, values)

        self._sentences_in_news = self._get_sentences_in_news()

        self.relation_collections[DataType.Train].debug_labels_statistic(DataType.Train)
        self.relation_collections[DataType.Train].debug_unique_relations_statistic()
        self.relation_collections[DataType.Test].debug_labels_statistic(DataType.Test)
        self.relation_collections[DataType.Test].debug_unique_relations_statistic()

    @property
    def Settings(self):
        return self._settings

    def _get_sentences_in_news(self):
        sentences_in_news = {}
        all = self.io.get_data_indices(DataType.Train) + self.io.get_data_indices(DataType.Test)
        for news_ID in all:
            sentences = News.read_sentences(io_utils.get_news_filepath(news_ID))
            sentences_in_news[news_ID] = len(sentences)
        return sentences_in_news

    def fit(self):
        assert(self.sess is not None)

        for epoch_index in range(self.Settings.Epochs):

            groups_count = 0
            total_cost = 0
            total_acc = 0

            self.relation_collections[DataType.Train].reset_labels()

            groups = list(self.relation_collections[DataType.Train].iter_by_linked_relations_groups(self.Settings.BatchSize))
            np.random.shuffle(groups)

            for index, relation_groups in enumerate(groups):

                feed_dict = self.create_feed_dict(Batch(relation_groups, self.Settings.GroupSize), DataType.Train)

                log_names, log_params = self.network.Log
                result = self.sess.run([self.optimiser, self.network.Cost, self.network.Accuracy] + log_params,
                                       feed_dict=feed_dict)
                total_cost += np.mean(result[1])
                total_acc += result[2]
                groups_count += 1

                if TextDebugKeys.FitBatchDisplayLog:
                    self._display_log(log_names, result[3:])

            if self.callback is not None:
                self.callback.on_epoch_finished(avg_cost=total_cost / groups_count,
                                                avg_acc=total_acc / groups_count,
                                                epoch_index=epoch_index)

        if self.callback is not None:
            self.callback.on_fit_finished()

    def predict(self, dest_data_type=DataType.Test):

        self.relation_collections[dest_data_type].reset_labels()

        for index, relation_groups in enumerate(self.relation_collections[dest_data_type].iter_by_linked_relations_groups(self.Settings.BatchSize)):

            batch = Batch(relation_groups, self.Settings.GroupSize)
            feed_dict = self.create_feed_dict(batch, dest_data_type)

            result = self.sess.run([self.network.Labels], feed_dict=feed_dict)
            uint_labels = result[0]

            for group_index, group in enumerate(batch.iter_groups):
                for relation in group:
                    assert(isinstance(relation, ExtractedRelation))
                    self.relation_collections[dest_data_type].apply_label(
                        label=Label.from_uint(int(uint_labels[group_index])),
                        relation_id=relation.relation_id)

        for news_ID in self.io.get_data_indices(dest_data_type):
            collection = OpinionCollection(None, self.synonyms, self.Settings.Stemmer)
            self.relation_collections[dest_data_type].fill_opinion_collection(collection,
                                                                              news_ID,
                                                                              lambda labels: labels[0],
                                                                              debug_check_collection=False)

            collection.save(self.io.get_opinion_output_filepath(news_ID, self.io.get_model_root(dest_data_type)))

        return self._evaluate(dest_data_type, self.Settings.Stemmer)

    def _set_optimiser(self):
        self.optimiser = self.Settings.Optimiser.minimize(self.network.Cost)

    def get_gpu_memory_fraction(self):
        return self.Settings.GPUMemoryFraction

    def create_feed_dict(self, batch, data_type):
        """
        returns: dict
            Returns dictionary for tensorflow session
        """
        assert(isinstance(batch, Batch))
        label_by_group_func = lambda group: self.relation_collections[data_type].get_original_label(group[0].relation_id)
        sentences_count_func = lambda news_id: self._sentences_in_news[news_id]
        input = batch.to_network_input(self.predicted_collections[data_type],
                                       label_by_group_func,
                                       sentences_count_func,
                                       self.Settings)
        return self.network.create_feed_dict(input, data_type)
