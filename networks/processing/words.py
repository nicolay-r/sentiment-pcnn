# -*- coding: utf-8 -*-

import numpy as np

from core.env import stemmer
from core.source.news import News
from core.source.entity import Entity
from core.runtime.embeddings import Embedding
from indices import EntityIndices


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

        # words + total amount of entities + 1 is reserved for unknown word
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

    def get_embedding_indices(self, news_ID, total_words_count, pad_size):
        assert(type(news_ID) == int)
        assert(type(pad_size) == int)
        return self.by_id[news_ID].to_embedding_indices(
            self.embedding,
            self.entity_indices,
            total_words_count,
            pad_size)

    def get_embedding_indices_in_window(self, news_ID, total_words_count, w_from, w_to):
        assert(type(news_ID) == int)
        assert(type(w_from) == int)
        assert(type(w_to) == int)
        return self.by_id[news_ID].to_embedding_indices(
            self.embedding,
            self.entity_indices,
            total_words_count,
            w_to - w_from,
            w_from, w_to)

    def get_pos_indices_in_window(self, news_ID, w_from, w_to):
        assert(type(w_from) == int)
        assert(type(w_to) == int)
        return self.by_id[news_ID].to_pos_indices(w_from, w_to)


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

    def to_pos_indices(self, w_from=None, w_to=None):
        """
        w_from: None or int
            position of initial word, is a beginning of a window in news
        w_from: None or int
            position of last word (not included), is an end of window in news

        returns: list int
            list of indices
        """
        assert(w_from is None or isinstance(w_from, int))
        assert(w_to is None or isinstance(w_to, int))

        indices = []
        if w_from is None and w_to is None:
            taken_words = self.words
        else:
            taken_words = self.words[w_from:w_to]

        unknown_index = len(stemmer.pos_names)

        for w in taken_words:
            if not isinstance(w, unicode):
                index = unknown_index
            else:
                index = stemmer.pos_to_int(stemmer.get_term_pos(w))
            indices.append(index)

        return indices

    def to_embedding_indices(self, embedding, entity_indices, total_words_count, pad_size,
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
        assert(isinstance(pad_size, int) and pad_size > 0)
        assert(w_from is None or isinstance(w_from, int))
        assert(w_to is None or isinstance(w_to, int))

        indices = []
        unknown_word_index = total_words_count - 1

        if w_from is None and w_to is None:
            taken_words = self.words
        else:
            taken_words = self.words[w_from:w_to]

        for w in taken_words:
            if isinstance(w, unicode):
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
