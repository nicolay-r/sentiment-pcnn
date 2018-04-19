from core.source.news import News
from words import NewsWords


class TextPosition:
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
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right


class NewsDescriptor:
    """ Contains necessary classes for a news
    """

    def __init__(self, news_ID, news, news_words, opinion_collections):
        assert(isinstance(news_ID, int))    # news index, which is a part of news filename
        assert(isinstance(news, News))
        assert(isinstance(news_words, NewsWords))
        assert(isinstance(opinion_collections, list))
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
