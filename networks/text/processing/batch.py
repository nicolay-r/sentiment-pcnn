import numpy as np
from networks.text.configurations.base import TextModelSettings, MergingSameSentenceMode
from networks.text.debug import TextDebugKeys
from networks.context.processing.predictions import RelationPredictionResultCollection
from networks.context.processing.extracted_relations import ExtractedRelation


class Batch:

    I_X = u"input_x"
    I_LABEL = u"label"
    I_EMBEDDING = u"embedding"
    I_SENTENCE_INDEX = u"sentence_index"

    def __init__(self, batch, group_size):
        assert(isinstance(batch, list))
        assert(isinstance(group_size, int))
        self.batch = batch
        self.group_size = group_size

    @property
    def iter_groups(self):
        for group in self.batch:
            yield group

    def to_network_input(self,
                         predicted_collection,
                         label_by_group_func,
                         sentences_in_news_func,
                         settings):
        """
        returns:
            x -- Embedding
            y -- output labels
            Embedding (where place output of context network)
        """
        assert(isinstance(predicted_collection, RelationPredictionResultCollection))
        assert(isinstance(settings, TextModelSettings))

        x = np.zeros((len(self.batch), self.group_size))
        labels = np.zeros(len(self.batch))
        predictions = np.zeros(settings.EmbeddingShape)
        sentence_indices = np.zeros((settings.EmbeddingShape[0], settings.TextPartsCount))

        for batch_index, group in enumerate(self.iter_groups):

            # Compose embedding rows, and grouped x
            if settings.MergingMode == MergingSameSentenceMode.NoMerge:
                for relation_index, relation in enumerate(group):
                    assert(isinstance(relation, ExtractedRelation))
                    assert(relation.text_position.left_entity_position.SentenceIndex ==
                           relation.text_position.right_entity_position.SentenceIndex)

                    if relation_index >= self.group_size:
                        break

                    predicted_relation = predicted_collection.get_by_relation_id(relation.relation_id)

                    sentences_count = sentences_in_news_func(relation.text_position.news_ID)
                    text_part = self.get_text_part_index(relation, sentences_count, settings)

                    row_index = batch_index * self.group_size + relation_index + 1
                    predictions[row_index] = predicted_relation.Prediction
                    sentence_indices[row_index][text_part] = 1
                    x[batch_index][relation_index] = row_index

            # TODO. Implement distributed placement with average for same sentence relation occurance
            if settings.MergingMode == MergingSameSentenceMode.Average:
                raise Exception("Not Supported")

            label = label_by_group_func(group)
            labels[batch_index] = label.to_uint()

        result = {Batch.I_X: x,
                  Batch.I_LABEL: labels,
                  Batch.I_EMBEDDING: predictions,
                  Batch.I_SENTENCE_INDEX: sentence_indices}

        if TextDebugKeys.BatchInfo:
            self.debug_show(result)

        return result

    @staticmethod
    def get_text_part_index(relation, sentences_count, settings):
        return int(settings.TextPartsCount * (1.0 *
                                              relation.text_position.left_entity_position.SentenceIndex /
                                              sentences_count))

    @staticmethod
    def debug_show(result):
        assert(isinstance(result, dict))
        print "-----------------------------"
        for key, value in result.iteritems():
            print "{}: {}".format(key, value)
        print "-----------------------------"

