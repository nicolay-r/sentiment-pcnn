# -*- coding: utf-8 -*-

from core.evaluation.labels import Label, PositiveLabel, NegativeLabel, NeutralLabel
from core.source.news import News
from core.source.opinion import OpinionCollection, Opinion
from core.source.synonyms import SynonymsCollection
from core.runtime.relations import RelationCollection, Relation
from batch import MiniBatch
from utils import TextPosition, NewsDescriptor


# TODO list:
# 1. Create a simple positions extractor for each relation (for both collections)
#       news_id => relations[]
#          relation => (pos_left, pos_right, label)
# 2. Implement expanded word2vec, which includes relation phrases and related
#    summaried vectors of phrases words. Quite important because of ENTITIES.
# 3. Implement news indexing based on expanded embedding of each news.
#       news_id => embedding_indices
# 4. Minibatch splitting
# 5. Feed network


class ExtractedRelationCollection:

    def __init__(self):
        self.relations = []         # list of tuples (positions, score)
        self.relation_values = []   # entity values  (left, right)

    # TODO. Duplicated from vectors.py
    @staticmethod
    def get_appropriate_entities(opinion_value, synonyms, entities):
        if synonyms.has_synonym(opinion_value):
            return filter(
                lambda s: entities.has_entity_by_value(s),
                synonyms.get_synonyms_list(opinion_value))
        elif entities.has_entity_by_value(opinion_value):
            return [opinion_value]
        else:
            return []

    @staticmethod
    def _in_window(relation, news_words, max_relation_words_width):
        assert(isinstance(relation, Relation))
        assert(type(max_relation_words_width) == int and max_relation_words_width >= 0)
        pos1 = ExtractedRelationCollection._get_entity_position(relation.entity_left_ID, news_words)
        pos2 = ExtractedRelationCollection._get_entity_position(relation.entity_right_ID, news_words)

        # we guarantee that window has a gap at both outer sides, [ ... e1 ... e2 ... ]
        #                                                            ^             ^
        if (abs(pos1 - pos2) + 4) > max_relation_words_width:
            return False

        return True

    @staticmethod
    def _get_distance_in_sentences(relation, news):
        assert(isinstance(relation, Relation))
        assert(isinstance(news, News))
        return abs(news.get_sentence_by_entity(relation.get_left_entity()).index -
                   news.get_sentence_by_entity(relation.get_right_entity()).index)

    @staticmethod
    def _get_entity_position(entity_ID, news_words):
        assert(type(entity_ID) == unicode)
        return news_words.get_entity_position(entity_ID)

    def add_news_relations(self,
                           news_descriptor,
                           synonyms_collection,
                           max_relation_words_width,
                           is_train_collection):
        assert(isinstance(news_descriptor, NewsDescriptor))
        assert(isinstance(synonyms_collection, SynonymsCollection))
        assert(type(max_relation_words_width) == int and max_relation_words_width >= 0)
        # Code the same as in vectors.py
        for opinions in news_descriptor.opinion_collections:
            assert(isinstance(opinions, OpinionCollection))
            for o in opinions:

                relations = RelationCollection.from_news_opinion(
                        news_descriptor.news, o, opinions.synonyms)

                if len(relations) == 0:
                    continue

                relations.apply_filter(lambda r: self._in_window(r, news_descriptor.news_words, max_relation_words_width))
                relations.apply_filter(lambda r: self._get_distance_in_sentences(r, news_descriptor.news) < 1)

                # TODO. For neutral labels (in case of test collection) we take
                # only IDs that occurs in the same sentence.
                if not is_train_collection and len(relations) > 0:
                    relations = [relations[0]]

                for r in relations:

                    pos1 = ExtractedRelationCollection._get_entity_position(
                        r.entity_left_ID, news_descriptor.news_words)
                    pos2 = ExtractedRelationCollection._get_entity_position(
                        r.entity_right_ID, news_descriptor.news_words)

                    self.relations.append(
                        (TextPosition(news_descriptor.news_index, pos1, pos2),
                         o.sentiment))

                    self.relation_values.append(
                        (r.get_left_entity_value(), r.get_right_entity_value()))

    def _find_relation_and_set_label(self, position, label):
        # O(N*2)
        assert(isinstance(position, TextPosition))
        assert(isinstance(label, Label))
        for index, r in enumerate(self.relations):
            p, _ = r
            assert(isinstance(p, TextPosition))
            if p.equals(position):
                self.relations[index] = (p, label)
                return
        raise Exception("Position was not found, ({}, {})".format(position.left, position.right))

    def apply_labels(self, uint_labels, minibatch):
        """
        uint_labels: list of int
            each label could be as follows: 0 -- neutral, and 1 -- positive, 2 -- negative

        Applying labels for each bag. It is supposed that labels and bags have
        the same order.
        """
        assert(isinstance(uint_labels, list))
        assert(isinstance(minibatch, MiniBatch))
        index = 0
        for bag in minibatch.bags:
            for position in bag.positions:
                label = Label.from_uint(uint_labels[index])
                self._find_relation_and_set_label(position, label)
                index += 1

    def to_opinion_collections(self, news_indices, synonyms):
        # O(N^2)
        assert(isinstance(news_indices, list))
        assert(isinstance(synonyms, SynonymsCollection))

        result = []
        for news_ID in news_indices:
            result_opinions = OpinionCollection(None, synonyms)

            for relation_index, r in enumerate(self.relations):

                p, label = r

                if p.news_ID != news_ID:
                    continue

                if label == NeutralLabel():  # ignore neutral labels
                    continue

                left_entity_value, right_entity_value = self.relation_values[relation_index]
                o = Opinion(left_entity_value, right_entity_value, label)
                if not result_opinions.has_opinion_by_synonyms(o):
                    result_opinions.add_opinion(o)

            result.append((news_ID, result_opinions))

        return result

    def debug_statistic(self):
        labels_count = [0, 0, 0]
        for _, label in self.relations:
            labels_count[label.to_uint()] += 1
        return {'pos': labels_count[PositiveLabel().to_uint()],
                'neg': labels_count[NegativeLabel().to_uint()],
                'neu': labels_count[NeutralLabel().to_uint()]}
