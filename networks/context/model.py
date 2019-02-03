import datetime
import numpy as np
import collections

from io import ContextLevelNetworkIO
from networks.callback import Callback

from core.runtime.relations import RelationCollection, Relation
from core.evaluation.labels import Label
from core.source.entity import EntityCollection
from core.source.synonyms import SynonymsCollection
from core.source.news import News
from core.source.opinion import OpinionCollection
from core.source.vectors import OpinionVectorCollection

from networks.io import DataType
from networks.model import TensorflowModel
from networks.network import NeuralNetwork
from networks.context.configurations.base import CommonModelSettings, LabelCalculationMode

from processing.sample import Sample
from processing.extracted_relations import ExtractedRelationsCollection
from processing.indices import EntityIndices
from processing.terms import NewsTerms, EntityPosition
from processing.batch import MiniBatch
from processing.bags import BagsCollection
from processing.terms import NewsTermsCollection
from processing.predictions import RelationPredictionResultCollection, RelationPredictionResult
from processing.utils import create_term_embedding_matrix

from debug import DebugKeys


class ContextLevelTensorflowModel(TensorflowModel):

    def __init__(self, io, settings, callback):
        assert(isinstance(io, ContextLevelNetworkIO))
        assert(isinstance(settings, CommonModelSettings))
        assert(isinstance(callback, Callback) or callback is None)
        super(ContextLevelTensorflowModel, self).__init__(io, callback)

        def iter_all_entity_collections():
            all_indices = io.get_data_indices(DataType.Train) + \
                          io.get_data_indices(DataType.Test)
            for news_index in all_indices:
                yield EntityCollection.from_file(self.io.get_entity_filepath(news_index),
                                                 self.Settings.Stemmer)

        self.settings = settings
        self._last_fit_epoch_index = None

        self.synonyms = SynonymsCollection.from_file(io.get_synonyms_collection_filepath(),
                                                     stemmer=self.Settings.Stemmer)

        self.all_existed_entity_indices = EntityIndices.from_entities_collections(
            iter_all_entity_collections(),
            self.Settings.Stemmer)

        # Train collection
        train_news_terms_collection, train_relations_collection = \
            self._process_into_collections(io.get_data_indices(DataType.Train), DataType.Train)

        # Test collection
        test_news_terms_collection, test_relations_collection = \
            self._process_into_collections(io.get_data_indices(DataType.Test), DataType.Test)

        self.Settings.update_terms_per_context(
            min(train_news_terms_collection.calculate_min_terms_per_context(),
                test_news_terms_collection.calculate_min_terms_per_context()))

        self.Settings.set_term_embedding(
            create_term_embedding_matrix(self.Settings.WordEmbedding, self.all_existed_entity_indices))

        self.bags_collection = {
            DataType.Test: BagsCollection(
                test_relations_collection,
                self.Settings.BagSize,
                shuffle=False,
                create_sample_func=lambda relation: Sample.from_relation(relation,
                                                                         self.all_existed_entity_indices,
                                                                         test_news_terms_collection,
                                                                         self.settings)),

            DataType.Train: BagsCollection(
                train_relations_collection,
                self.Settings.BagSize,
                shuffle=True,
                create_sample_func=lambda relation: Sample.from_relation(relation,
                                                                         self.all_existed_entity_indices,
                                                                         train_news_terms_collection,
                                                                         self.settings))
        }

        self._relations_collections = {
            DataType.Test: test_relations_collection,
            DataType.Train: train_relations_collection
        }

        self.news_terms_collections = {
            DataType.Test: test_news_terms_collection,
            DataType.Train: train_news_terms_collection
        }

        neu, pos, neg = self._relations_collections[DataType.Train].get_statistic()
        self.settings.set_class_weights([100.0 / neu, 100.0 / pos, 100.0 / neg])

        # log
        keys, values = self.settings.get_parameters()
        self._display_log(keys, values)

        self._relations_collections[DataType.Train].debug_labels_statistic(DataType.Train)
        self._relations_collections[DataType.Train].debug_unique_relations_statistic()
        self._relations_collections[DataType.Test].debug_labels_statistic(DataType.Test)
        self._relations_collections[DataType.Test].debug_unique_relations_statistic()

    @property
    def RelationsCollections(self):
        return self._relations_collections

    @property
    def Settings(self):
        return self.settings

    @property
    def AllExistedEntityIndices(self):
        return self.all_existed_entity_indices

    def get_gpu_memory_fraction(self):
        return self.Settings.GPUMemoryFraction

    def fit(self):
        assert(self.sess is not None)

        minibatches = list(self.bags_collection[DataType.Train].iter_by_groups(self.Settings.BagsPerMinibatch))

        for epoch_index in range(self.Settings.Epochs):

            if DebugKeys.FitTrainCollectionLabelsInfo:
                self._relations_collections[DataType.Train].debug_labels_statistic(DataType.Train)

            total_cost = 0
            total_acc = 0
            groups_count = 0

            np.random.shuffle(minibatches)

            for bags_group in minibatches:

                feed_dict = self.create_feed_dict(MiniBatch(bags_group), data_type=DataType.Train)

                log_names, log_params = self.network.Log
                result = self.sess.run([self.optimiser, self.network.Cost, self.network.Accuracy] + log_params,
                                        feed_dict=feed_dict)
                cost = result[1]

                if DebugKeys.FitBatchDisplayLog:
                    self._display_log(log_names, result[3:])

                total_cost += np.mean(cost)
                total_acc += result[2]
                groups_count += 1

            self.save_model(self.io.get_model_state_filepath(epoch_index))

            self._last_fit_epoch_index = epoch_index

            if self.callback is not None:
                self.callback.on_epoch_finished(avg_cost=total_cost / groups_count,
                                                avg_acc=total_acc / groups_count,
                                                epoch_index=epoch_index)

        if self.callback is not None:
            self.callback.on_fit_finished()

    def _process_into_collections(self, indices, data_type):
        """
        Processing all parameters into collections.
        returns:
            NewsWordsCollection and RelationCollection
        """
        def find_feature_vector_for_opinion(opinion_vector_collections, opinion):
            assert(isinstance(opinion_vector_collections, list))

            for collection in opinion_vector_collections:
                assert(isinstance(collection, OpinionVectorCollection))
                if not collection.has_opinion(opinion):
                    continue
                return collection.find_by_opinion(opinion)

            return None

        assert(isinstance(indices, list))

        erc = ExtractedRelationsCollection()
        ntc = NewsTermsCollection()
        for news_index in indices:
            assert(isinstance(news_index, int))

            entity_filepath = self.io.get_entity_filepath(news_index)
            news_filepath = self.io.get_news_filepath(news_index)
            opin_filepath = self.io.get_opinion_input_filepath(news_index)
            neutral_filepath = self.io.get_neutral_filepath(news_index, data_type)

            news = News.from_file(news_filepath,
                                  EntityCollection.from_file(entity_filepath, self.Settings.Stemmer),
                                  stemmer=self.Settings.Stemmer)

            opinions_collections = [OpinionCollection.from_file(neutral_filepath,
                                                                self.io.get_synonyms_collection_filepath(),
                                                                self.Settings.Stemmer)]
            if data_type == DataType.Train:
                opinions_collections.append(OpinionCollection.from_file(opin_filepath,
                                                                        self.io.get_synonyms_collection_filepath(),
                                                                        self.Settings.Stemmer))

            news_terms = NewsTerms.create_from_news(news_index, news, keep_tokens=self.Settings.KeepTokens)

            for relations, opinion in self._extract_relations(opinions_collections, news, news_terms):

                feature_vector = find_feature_vector_for_opinion(self.get_opinion_vector_collection(news_index, data_type),
                                                                 opinion)

                erc.add_news_relations(relations,
                                       opinion,
                                       news_terms,
                                       news_index,
                                       feature_vector)
            ntc.add_news_terms(news_terms)

        return ntc, erc

    def _extract_relations(self, opinion_collections, news, news_terms):
        assert(isinstance(opinion_collections, collections.Iterable))
        assert(isinstance(news, News))
        assert(isinstance(news_terms, NewsTerms))

        def filter_by_terms_per_context(relation):
            assert(isinstance(relation, Relation))
            assert(self.Settings.TermsPerContext is not None)
            pos1 = news_terms.get_entity_position(relation.entity_left_ID)
            pos2 = news_terms.get_entity_position(relation.entity_right_ID)
            assert(isinstance(pos1, EntityPosition) and isinstance(pos2, EntityPosition))

            # we guarantee that window has a gap at both outer sides, [ ... e1 ... e2 ... ]
            #                                                            ^             ^
            if (abs(pos1.TermIndex - pos2.TermIndex) + 4) > self.Settings.TermsPerContext:
                return False

            return True

        def filter_by_distance_in_sentences(relation):
            return abs(news.get_sentence_by_entity(relation.get_left_entity()).index -
                       news.get_sentence_by_entity(relation.get_right_entity()).index)

        for opinions in opinion_collections:
            assert (isinstance(opinions, OpinionCollection))
            for opinion in opinions:

                relations = RelationCollection.from_news_opinion(news, opinion, opinions.synonyms)

                if len(relations) == 0:
                    continue

                relations.apply_filter(lambda relation: filter_by_terms_per_context(relation))
                relations.apply_filter(lambda relation: filter_by_distance_in_sentences(relation) == 0)

                yield relations, opinion

    def predict(self, dest_data_type=DataType.Test):

        def calculate_label(relation_labels):
            assert(isinstance(relation_labels, list))

            label = None
            if self.Settings.RelationLabelCalculationMode == LabelCalculationMode.FIRST_APPEARED:
                label = relation_labels[0]
            if self.Settings.RelationLabelCalculationMode == LabelCalculationMode.AVERAGE:
                label = Label.from_int(np.sign(sum([l.to_int() for l in relation_labels])))

            if DebugKeys.PredictLabel:
                print [l.to_int() for l in relation_labels]
                print "Result: {}".format(label.to_int())

            return label

        assert(isinstance(dest_data_type, unicode))

        self._relations_collections[dest_data_type].reset_labels()
        prediction_collection = RelationPredictionResultCollection(len(self._relations_collections[dest_data_type]))

        for bags_group in self.bags_collection[dest_data_type].iter_by_groups(self.Settings.BagsPerMinibatch):

            minibatch = MiniBatch(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=dest_data_type)

            log_names, log_params = self.network.Log
            result = self.sess.run([self.network.Labels, self.network.Output] + log_params, feed_dict=feed_dict)
            uint_labels = result[0]
            output = result[1]

            if DebugKeys.PredictBatchDisplayLog:
                self._display_log(log_names, result[2:])

            # apply labels
            sample_indices_count = 0
            for sample_index, sample in enumerate(minibatch.iter_by_samples()):
                label = Label.from_uint(int(uint_labels[sample_index]))
                self._relations_collections[dest_data_type].apply_label(label, sample.RelationID)
                prediction_collection.add(sample.RelationID, RelationPredictionResult(output[sample_index]))
                sample_indices_count += 1

            assert(sample_indices_count == len(uint_labels))

        assert(self._relations_collections[dest_data_type].debug_check_all_relations_has_labels())

        self._relations_collections[dest_data_type].debug_labels_statistic(dest_data_type)

        # Compose Result
        self._relations_collections[dest_data_type].save(
            self.io.get_relations_filepath(data_type=dest_data_type,
                                           epoch=self._last_fit_epoch_index))

        prediction_collection.save(
            self.io.get_relations_prediction_filepath(data_type=dest_data_type,
                                                      epoch=self._last_fit_epoch_index))

        for news_ID in self.io.get_data_indices(dest_data_type):
            collection = OpinionCollection(None, self.synonyms, self.settings.Stemmer)
            self._relations_collections[dest_data_type].fill_opinion_collection(collection, news_ID, calculate_label)

            collection.save(self.io.get_opinion_output_filepath(news_ID, self.io.get_model_root(dest_data_type)))

        return self._evaluate(dest_data_type, self.Settings.Stemmer)

    def _set_optimiser(self):
        self.optimiser = self.Settings.Optimiser.minimize(self.network.Cost)

    def get_opinion_vector_collection(self, news_ID, data_type):
        return []

    def create_feed_dict(self, mbatch, data_type):
        assert(isinstance(self.network, NeuralNetwork))
        assert(isinstance(data_type, unicode))
        input = mbatch.to_network_input()

        if DebugKeys.FeedDictShow:
            MiniBatch.debug_output(input)

        return self.network.create_feed_dict(input, data_type)
