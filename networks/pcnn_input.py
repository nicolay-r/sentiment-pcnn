# -*- coding: utf-8 -*-

import numpy as np
from gensim.models.word2vec import Word2Vec

import io_utils
from core.env import stemmer
from core.evaluation.labels import Label, PositiveLabel, NegativeLabel, NeutralLabel
from core.source.entity import EntityCollection, Entity
from core.source.news import News
from core.source.opinion import OpinionCollection, Opinion
from core.source.synonyms import SynonymsCollection
from core.runtime.relations import RelationCollection, Relation
from core.runtime.embeddings import Embedding


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


class EntityIndices:
    """
    Collects all entities from multiple entities_collections
    """

    def __init__(self, entities_collections):
        assert(type(entities_collections) == list)
        entity_index = 0
        self.e_value = {}
        self.e_index = {}
        for entities in entities_collections:
            assert(isinstance(entities, EntityCollection))
            for e in entities:
                assert(isinstance(e, Entity))
                terms = self.value_to_terms(e.value)
                k = self._terms_to_key(terms)
                if k not in self.e_index:
                    self.e_index[k] = entity_index
                    self.e_value[entity_index] = e.value
                    entity_index += 1

    @staticmethod
    def value_to_terms(entity_value):
        assert(type(entity_value) == unicode)
        return stemmer.lemmatize_to_list(entity_value)

    @staticmethod
    def _terms_to_key(terms):
        return '_'.join(terms)

    def get_entity_index(self, entity):
        """
        returns: int or None
            index according to the collection this collection, or None in case
            of undefined 'entity'.
        """
        assert(isinstance(entity, Entity))
        terms = self.value_to_terms(entity.value)
        key = self._terms_to_key(terms)
        if key in self.e_index:
            return self.e_index[key]
        return None

    def iter_entity_index_and_values(self):
        for e_index, e_value in self.e_value.items():
            yield (e_index, e_value)

    def __len__(self):
        return len(self.e_index)


class NewsWordsCollection:

    def __init__(self, entity_indices, embedding):
        # entites_embedding and w2v actualy used only in case of indices
        # extraction from news
        assert(isinstance(entity_indices, EntityIndices))
        assert(isinstance(embedding, Embedding))
        self.entity_indices = entity_indices
        self.embedding = embedding
        self.by_id = {}

    def add_news(self, news_words):
        assert(isinstance(news_words, NewsWords))
        assert(news_words.ID not in self.by_id)
        self.by_id[news_words.ID] = news_words

    def get_embedding_matrix(self):
        """
        returns: np.ndarray(words_count, embedding_size)
            embedding matrix which includes embedding both for words and
            entities
        """
        w2v_words_count = len(self.embedding.vocab)
        entities_count = len(self.entity_indices)
        vector_size = self.embedding.vector_size

        # words + total amount of enttities + 1 is reserved for unknown word
        matrix = np.zeros((w2v_words_count + entities_count + 1, vector_size))

        for word, info in self.embedding.vocab.items():
            index = info.index
            matrix[index] = self.embedding.get_vector_by_index(index)

        for e_index, e_value in self.entity_indices.iter_entity_index_and_values():
            terms = self.entity_indices.value_to_terms(e_value)
            matrix[w2v_words_count + e_index] = self._get_mean_word2vec_vector(terms)

        return matrix

    def _get_mean_word2vec_vector(self, terms):
        v = np.zeros(self.embedding.vector_size, dtype=np.float32)
        for term in terms:
            if term in self.embedding:
                v = v + self.embedding[term]
        return v

    def get_words_per_news(self, news_ID):
        return len(self.by_id[news_ID])

    def get_min_words_per_news_count(self):
        result = None
        for news_words in self.by_id.itervalues():
            if result is None:
                result = len(news_words)
            else:
                result = min(len(news_words), result)
        return result

    def get_indices(self, news_ID, total_words_count, pad_size):
        assert(type(news_ID) == int)
        assert(type(pad_size) == int)
        return self.by_id[news_ID].to_indices(
                self.embedding,
                self.entity_indices,
                total_words_count,
                pad_size)

    def get_indices_in_window(self, news_ID, total_words_count, w_from, w_to):
        assert(type(news_ID) == int)
        assert(type(w_from) == int)
        assert(type(w_to) == int)
        return self.by_id[news_ID].to_indices(
                self.embedding,
                self.entity_indices,
                total_words_count,
                w_to - w_from,
                w_from, w_to)


class NewsWords:

    GARBAGE = [u',', u':', u'}', u'{', u'[', u']', u')', u'(', u'.', u'-',
               u'«', u'•', u'?', u'!', u'»', u'».']

    DIGITS = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9']

    def __init__(self, news_ID, news):
        assert(type(news_ID) == int)
        assert(isinstance(news, News))
        words, positions = self._split_by_words_and_entities(news)
        self.news_ID = news_ID
        self.words = words
        self.entity_positions = positions

    @property
    def ID(self):
        return self.news_ID

    # TODO. move this method into the kernel (at entity.py file).
    def _split_by_words_and_entities(self, news):
        words = []
        entity_position = {}
        for s in news.sentences:
            s_pos = s.begin
            for e_ID, e_begin, e_end in s.entity_info:
                # add words before entity
                if e_begin > s_pos:
                    words.extend(self.split_text(s.text[s_pos:e_begin]))
                # add entity position
                entity_position[e_ID] = len(words)
                # add entity_text
                words.append(news.entities.get_entity_by_id(e_ID))
                s_pos = e_end
            # add text part after last entity of sentence.
            words.extend(self.split_text(s.text[s_pos:s.end]))

        return words, entity_position

    def get_entity_position(self, e_ID):
        assert(type(e_ID) == unicode)  # ID which is a part of *.ann files.
        return self.entity_positions[e_ID]

    @staticmethod
    def split_text(text):
        splitted = [w.strip() for w in text.split(' ')]
        for c in NewsWords.GARBAGE:
            if c in splitted:
                splitted.remove(c)

        # ignore words that include digits
        for d in NewsWords.DIGITS:
            for c in splitted:
                if d in c:
                    splitted.remove(c)

        splitted = filter(None, splitted)

        # for s in splitted:
        #     print '"{}" '.format(s.encode('utf-8')),

        return splitted

    def to_indices(self, embedding, entity_indices, total_words_count, pad_size,
                   w_from=None, w_to=None):
        """
        w_from: None or int
            position of initial word, is a beginning of a window in news
        w_from: None or int
            position of last word (not included), is an end of window in news

        returns: list int
            list of indices
        """
        # O(N^2) because of search at embedding by word to obtain related
        # index.
        assert(isinstance(embedding, Embedding))
        assert(isinstance(entity_indices, EntityIndices))
        assert(type(pad_size) == int and pad_size > 0)
        assert(type(w_from) == None or type(w_from) == int)
        assert(type(w_to) == None or type(w_to) == int)

        indices = []
        unknown_word_index = total_words_count - 1

        if (w_from is None and w_to is None):
            taken_words = self.words
        else:
            taken_words = self.words[w_from:w_to]

        for w in taken_words:
            if type(w) == unicode:
                terms = stemmer.lemmatize_to_list(w)
                if len(terms) == 0:
                    index = unknown_word_index
                else:
                    term = terms[0]
                    if term in embedding:
                        index = embedding.find_index_by_word(term) # TODO. O(N) now, search
                    else:
                        index = unknown_word_index
            elif isinstance(w, Entity):
                e = w
                index = entity_indices.get_entity_index(e) + len(embedding.vocab)
            else:
                raise Exception("Unsuported type {}".format(w))

            indices.append(index)

        assert(pad_size - len(indices) >= 0)
        indices.extend([unknown_word_index] * (pad_size - len(indices)))

        return indices

    def debug_show(self):
        for w in self.words:
            if type(w) == unicode:
                print "-> {}    '{}'".format(len(w), w.encode('utf-8'))
            elif isinstance(w, Entity):
                print "Entiity: '{}'".format(w.value)
            else:
                raise Exception("unsuported type {}".format(w))

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        for w in self.words_and_entities_list:
            yield w


class ExtractedRelationCollection:

    def __init__(self):
        self.relations = []         # list of tuples (positions, score)
        self.relation_values = []   # entity values  (left, right)

    # TODO. Duplicated from vectors.py
    @staticmethod
    def is_ignored(entity_value):
        ignored = io_utils.get_ignored_entity_values()
        entity_value = stemmer.lemmatize_to_str(entity_value)
        if entity_value in ignored:
            # print "ignored: '{}'".format(entity_value.encode('utf-8'))
            return True
        return False

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
        assert(type(uint_labels) == list)
        assert(isinstance(minibatch, MiniBatch))
        index = 0
        for bag in minibatch.bags:
            for position in bag.positions:
                label = Label.from_uint(uint_labels[index])
                self._find_relation_and_set_label(position, label)
                index += 1

    def to_opinion_collections(self, news_indices, synonyms):
        # O(N^2)
        assert(type(news_indices) == list)
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



class MiniBatch:

    def __init__(self, bags):
        assert(type(bags) == list)
        self.bags = bags

    @staticmethod
    def _dist(pos, size):
        result = []
        for i in range(size):
            result.append(i-pos if i-pos >= 0 else i-pos+size)
        return result

    def shuffle(self):
        np.random.shuffle(self.bags)

    @staticmethod
    def _get_related_to_window_entities_positions(window_size, left, right, words_in_news):
        """
        returns: tuple
            related left, and related right relation posiitions and window
            bounds as [w_from, w_to)
        """
        assert(type(left) == int and type(right) == int)
        assert(left != right)
        assert(abs(left - right) <= window_size)
        assert(window_size <= words_in_news)
        # ...outer_left... ENTITY1 ...inner... ENTITY2 ...outer_right...

        a_left = min(left, right)   # actual left
        a_right = max(left, right)  # actual right

        inner_size = a_right - a_left - 1
        outer_size = window_size - inner_size - 2
        outer_left_size = min(outer_size / 2, a_left)
        outer_right_size = window_size - outer_left_size - (2 + inner_size)

        w_from = a_left - outer_left_size
        w_to = a_right + outer_right_size + 1

        return left - w_from, right - w_from, w_from, w_to

    def to_network_input(self, news_collection, total_words_count, news_window_size):
        """
        news_window_size: int
            max amount of words per relation in news, i.e. a window.
        total_words_count: int
            amount of existed words (in embedding dictionary especially).
        returns: (X, p1, p2, P1, P2, y)

        Note: 'y' has an unsigned labels
        """
        assert(isinstance(news_collection, NewsWordsCollection))
        X = []
        p1 = []
        p2 = []
        P1 = []
        P2 = []
        y = []

        for b in self.bags:
            assert(isinstance(b, Bag))
            for position in b:

                l, r, w_from, w_to = self._get_related_to_window_entities_positions(
                    news_window_size,
                    position.left, position.right,
                    news_collection.get_words_per_news(position.news_ID))

                # check that there is a gap by both outer sides of a relation.
                assert((l > 0) and (r + w_from < w_to))

                indices = news_collection.get_indices_in_window(
                    position.news_ID, total_words_count, w_from, w_to)

                X.append(indices)
                p1.append(l)
                p2.append(r)
                P1.append(self._dist(l, news_window_size))
                P2.append(self._dist(r, news_window_size))
                y.append(b.label.to_uint())

                # self.debug_show(news_collection, indices, w_from, w_to, l, r,
                #     total_words_count)

        return (X, p1, p2, P1, P2, y)

    def debug_show(self, news_collection, indices, w_from, w_to, l, r, total_words_count):
        assert(isinstance(news_collection, NewsWordsCollection))
        for p, i in enumerate(indices):
            if p == l:
                print "__LEFT__",
            if p == r:
                print "__RIGHT__",
            if i < len(news_collection.embedding.vocab):
                print news_collection.embedding.get_vector_by_index(i),
            elif i < total_words_count - 1:
                print "__entity__",
            else:
                print "U",
        print '\n'

class BagsCollection:

    def __init__(self, relations, bag_size=40):
        assert(type(bag_size) == int and bag_size > 0)  # relations from relationsCollection
        assert(type(relations) == list)  # relations from relationsCollection

        self.bags = []
        self.bag_size = bag_size
        pos_bag_ind = self._add_bag(PositiveLabel())
        neg_bag_ind = self._add_bag(NegativeLabel())
        neu_bag_ind = self._add_bag(NeutralLabel())

        for position, label in relations:  # TODO. Shuffle
            assert(isinstance(position, TextPosition))
            assert(isinstance(label, Label))

            self._optional_add_in_bag(pos_bag_ind, position, label)
            self._optional_add_in_bag(neg_bag_ind, position, label)
            self._optional_add_in_bag(neu_bag_ind, position, label)

            if len(self.bags[pos_bag_ind]) == bag_size:
                pos_bag_ind = self._add_bag(PositiveLabel())
            if len(self.bags[neg_bag_ind]) == bag_size:
                neg_bag_ind = self._add_bag(NegativeLabel())
            if len(self.bags[neu_bag_ind]) == bag_size:
                neu_bag_ind = self._add_bag(NeutralLabel())

    def _add_bag(self, label):
        assert(isinstance(label, Label))
        self.bags.append(Bag(label))
        return len(self.bags) - 1

    def _optional_add_in_bag(self, bag_index, position, label):
        bag = self.bags[bag_index]
        if bag.label == label:
            bag.add_position(position)

    def shuffle(self):
        np.random.shuffle(self.bags)

    def to_minibatches(self, bags_per_minibatch=50):
        """
        returns: list of MiniBatch
        """
        # Note: It cuts the last bag because of strict size of each
        # minibatch and of each bag that becomes a part of minibatch.
        assert(type(bags_per_minibatch) == int and bags_per_minibatch > 0)
        filtered_bags = [b for b in self.bags if len(b) == self.bag_size]
        minibatches_count = len(filtered_bags) / bags_per_minibatch
        cutted = filtered_bags[:minibatches_count * bags_per_minibatch]
        grouped_bags = np.reshape(cutted, [minibatches_count, bags_per_minibatch])
        return [MiniBatch(bags.tolist()) for bags in grouped_bags]


class Bag:
    """
    Bag is a list of positions grouped by 'label'. So each position of
    realtion in dataset has the same 'label'
    """

    def __init__(self, label):
        assert(isinstance(label, Label))
        self.positions = []
        self._label = label

    def add_position(self, position):
        assert(isinstance(position, TextPosition))
        self.positions.append(position)

    def __len__(self):
        return len(self.positions)

    def __iter__(self):
        for p in self.positions:
            yield p

    @property
    def label(self):
        return self._label


class TextPosition:
    def __init__(self, news_ID, left, right):
        assert(type(news_ID) == int)    # news index, which is a part of news filename
        assert(type(left) == int)
        assert(type(right) == int)
        self._news_ID = news_ID
        self._left = left
        self._right = right

    def equals(self, other):
        assert(isinstance(other, TextPosition))
        return self._left == other._left \
               and self._right == other._right \
               and self._news_ID == other._news_ID

    @property
    def news_ID(self):
        return self._news_ID

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right


class NewsDescriptor:
    """ Contains necessary classes for a news
    """

    def __init__(self, news_ID, news, news_words, opinion_collections):
        assert(type(news_ID) == int)    # news index, which is a part of news filename
        assert(isinstance(news, News))
        assert(isinstance(news_words, NewsWords))
        assert(type(opinion_collections) == list)
        self._news_ID = news_ID
        self._news = news
        self._news_words = news_words
        self._opinion_collections = opinion_collections

    @property
    def opinion_collections(self):
        return self._opinion_collections

    @property
    def news_index(self):
        return self._news_ID

    @property
    def news(self):
        return self._news

    @property
    def entities(self):
        return self._news.entities

    @property
    def news_words(self):
        return self._news_words
