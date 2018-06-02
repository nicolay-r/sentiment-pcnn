from core.evaluation.labels import Label
from core.source.opinion import Opinion
from core.source.vectors import OpinionVectorCollection, OpinionVector
from core.source.news import News

from words import NewsWords


class TextPosition:
    """
    Represents an article sample by given newsID,
    and [left, right] entitiues positions
    """

    def __init__(self, news_ID, left, right):
        assert(isinstance(news_ID, int))    # news index, which is a part of news filename
        assert(isinstance(left, int))
        assert(isinstance(right, int))
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
    def left_entity_index(self):
        return self._left

    @property
    def right_entity_index(self):
        return self._right


class NewsDescriptor:
    """
    Contains necessary classes for a news
    """

    def __init__(self, news_ID, news, news_words, opinion_collections, opinion_vectors_collections):
        assert(isinstance(news_ID, int))    # news index, which is a part of news filename
        assert(isinstance(news, News))
        assert(isinstance(news_words, NewsWords))
        assert(isinstance(opinion_collections, list))
        assert(isinstance(opinion_vectors_collections, list))
        self._news_ID = news_ID
        self._news = news
        self._news_words = news_words
        self._opinion_collections = opinion_collections
        self._opinion_vectors_collections = opinion_vectors_collections

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

    @property
    def opinion_vector_collections(self):
        return self._opinion_vectors_collections


class ExtractedRelation:
    """
    Represents a relation which were found in news article
    and composed between two named entities
    (it was found especially by Opinion with predefined label)
    """

    def __init__(self, opinion_vector, text_position, left_entity_value, right_entity_value, label):
        assert(isinstance(opinion_vector, OpinionVector) or opinion_vector is None)
        assert(isinstance(text_position, TextPosition))
        assert(isinstance(left_entity_value, unicode))
        assert(isinstance(right_entity_value, unicode))
        assert(isinstance(label, Label))
        self.opinion_vector = opinion_vector  # NLP vector
        self.text_position = text_position
        self.left_entity_value = left_entity_value
        self.right_entity_value = right_entity_value
        self.label = label

    def create_opinion(self):
        return Opinion(self.left_entity_value, self.right_entity_value, self.label)
