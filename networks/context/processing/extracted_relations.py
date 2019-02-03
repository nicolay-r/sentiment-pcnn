# -*- coding: utf-8 -*-
import cPickle as pickle

from core.runtime.relations import RelationCollection
from core.evaluation.labels import Label, NeutralLabel, PositiveLabel, NegativeLabel
from core.source.opinion import OpinionCollection, Opinion
from core.source.vectors import OpinionVector

from networks.context.processing.terms import NewsTerms, EntityPosition


class ExtractedRelationsCollection:
    """
    Describes text relations with a position precision and forward connection, so
    for each relation we store it's continuation if it originally has.

    Usage:
    It is possible to add relations from news via add_news_relations

    Limitations:
    Not it represents IN-MEMORY implementation.
    Therefore it is supposed not so large amount of relations.
    """

    NO_NEXT_RELATION = None

    def __init__(self):
        # list ExtractedRelations
        self.relations = []
        # list describes that has i'th relation continuation in text.
        self.next_relation_id = []
        # provides original label by relation_id
        self.opinion_labels = []
        # labels defined
        self.debug_labels_defined = []

    def add_news_relations(self,
                           relations,
                           opinion,
                           news_terms,
                           news_index,
                           feature_vector):
        assert(isinstance(relations, RelationCollection))
        assert(isinstance(opinion, Opinion))
        assert(isinstance(news_terms, NewsTerms))
        assert(isinstance(news_index, int))

        for index, relation in enumerate(relations):

            pos_subj = news_terms.get_entity_position(relation.entity_left_ID)
            pos_obj = news_terms.get_entity_position(relation.entity_right_ID)
            relation_id = len(self.relations)

            extracted_relation = ExtractedRelation(
                feature_vector,
                ExtractedRelationPosition(news_index, pos_subj, pos_obj),
                relation_id,
                relation.get_left_entity_value(),
                relation.get_right_entity_value(),
                opinion.sentiment)

            self.relations.append(extracted_relation)
            self.next_relation_id.append(relation_id + 1 if index < len(relations) - 1 else self.NO_NEXT_RELATION)
            self.opinion_labels.append(opinion.sentiment)
            self.debug_labels_defined.append(True)

    def apply_label(self, label, relation_id):
        assert(isinstance(label, Label))
        self.relations[relation_id].set_label(label)
        self.debug_labels_defined[relation_id] = True

    def fill_opinion_collection(self, opinions, news_ID, calculate_label_func, debug_check_collection=True):
        assert(isinstance(opinions, OpinionCollection) and len(opinions) == 0)
        assert(isinstance(news_ID, int))

        for linked_relations in self._iter_by_linked_relations():

            first_relation = linked_relations[0]

            if first_relation.text_position.news_ID != news_ID:
                continue

            opinion = opinions.create_opinion(value_left=first_relation.left_entity_value,
                                              value_right=first_relation.right_entity_value,
                                              sentiment=calculate_label_func([r.label for r in linked_relations]))

            assert(isinstance(opinion, Opinion))
            if isinstance(opinion.sentiment, NeutralLabel):
                continue

            if opinions.has_opinion_by_values(opinion):
                continue

            opinions.add_opinion(opinion)

            if debug_check_collection:
                self._debug_check_relations_in_opinion_collection(linked_relations, opinions)

        return opinions

    def get_original_label(self, relation_id):
        assert(isinstance(relation_id, int))
        return self.opinion_labels[relation_id]

    def reset_labels(self):
        """
        Restores all labels that were assigned, when collection were
        implemented first time
        """
        for relation in self.relations:
            relation.label = self.opinion_labels[relation.relation_id]

        self.debug_labels_defined = [False] * len(self.debug_labels_defined)

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    @classmethod
    def load(cls, pickle_filepath):
        return pickle.load(open(pickle_filepath, 'rb'))

    def __iter__(self):
        for relation in self.relations:
            yield relation

    def __len__(self):
        return len(self.relations)

    def _iter_by_linked_relations(self):
        lst = []
        for index, relation in enumerate(self.relations):
            lst.append(relation)
            if self.next_relation_id[index] == self.NO_NEXT_RELATION:
                yield lst
                lst = []

    def iter_by_linked_relations_groups(self, group_size):
        assert(isinstance(group_size, int))
        group = []
        for index, linked_relations in enumerate(self._iter_by_linked_relations()):
            group.append(linked_relations)
            if len(group) == group_size:
                yield group
                group = []

    def get_statistic(self):
        neu = filter(lambda r: isinstance(r.label, NeutralLabel), self.relations)
        pos = filter(lambda r: isinstance(r.label, PositiveLabel), self.relations)
        neg = filter(lambda r: isinstance(r.label, NegativeLabel), self.relations)
        return 100.0 * len(neu) / len(self.relations), \
               100.0 * len(pos) / len(self.relations), \
               100.0 * len(neg) / len(self.relations)

    def _get_group_statistic(self):
        statistic = {}
        for group in self._iter_by_linked_relations():
            key = len(group)
            if key not in statistic:
                statistic[key] = 1
            else:
                statistic[key] += 1
        return statistic

    def _debug_check_relations_in_opinion_collection(self, relations, collection):
        assert(isinstance(relations, list))
        assert(isinstance(collection, OpinionCollection))

        for r in relations:
            assert(isinstance(r, ExtractedRelation))
            opinion = collection.create_opinion(value_left=r.left_entity_value,
                                                value_right=r.right_entity_value,
                                                sentiment=NeutralLabel())
            assert(collection.has_opinion_by_synonyms(opinion))

    def debug_labels_statistic(self, data_type):
        assert(isinstance(data_type, unicode))
        neu, pos, neg = self.get_statistic()
        print "Extracted relation collection: {}".format(data_type)
        print "\tTotal: {}".format(len(self.relations))
        print "\tNEU: {}%".format(neu)
        print "\tPOS: {}%".format(pos)
        print "\tNEG: {}%".format(neg)

    def debug_unique_relations_statistic(self):
        statistic = self._get_group_statistic()
        total = sum(list(statistic.itervalues()))
        print "Unique relations statistic:"
        print "\tTotal: {}".format(total)
        for key, value in sorted(statistic.iteritems()):
            print "\t{} -- {} ({}%)".format(key, value, 100.0 * value / total)
            total += value

    def debug_check_all_relations_has_labels(self):
        return not (False in self.debug_labels_defined)

class ExtractedRelation:
    """
    Represents a relation which were found in news article
    and composed between two named entities
    (it was found especially by Opinion with predefined label)
    """

    def __init__(self, opinion_vector,
                 text_position,
                 relation_id,
                 left_entity_value,
                 right_entity_value,
                 label):
        assert(isinstance(opinion_vector, OpinionVector) or opinion_vector is None)
        assert(isinstance(text_position, ExtractedRelationPosition))
        assert(isinstance(relation_id, int))
        assert(isinstance(left_entity_value, unicode))
        assert(isinstance(right_entity_value, unicode))
        assert(isinstance(label, Label))
        self.opinion_vector = opinion_vector  # NLP vector
        self.text_position = text_position
        self.relation_id = relation_id
        self.left_entity_value = left_entity_value
        self.right_entity_value = right_entity_value
        self.label = label

    def set_label(self, label):
        assert(isinstance(label, Label))
        self.label = label


class ExtractedRelationPosition:
    """
    Represents an article sample by given newsID,
    and [left, right] entities positions
    """

    def __init__(self, news_ID, left, right):
        assert(isinstance(news_ID, int))    # news index, which is a part of news filename
        assert(isinstance(left, EntityPosition))
        assert(isinstance(right, EntityPosition))
        self._news_ID = news_ID
        self._left = left
        self._right = right

    @property
    def news_ID(self):
        return self._news_ID

    @property
    def left_entity_position(self):
        return self._left

    @property
    def right_entity_position(self):
        return self._right
