from networks.processing.words import NewsWordsCollection
from networks.processing.utils import TextPosition


class Sample(object):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    def __init__(self, position):
        assert(isinstance(position, TextPosition))
        self.position = position

    def to_network_input(self, news_collection, news_window_size, total_words_count):
        """
        total_words_count: int
            amount of existed words (in embedding dictionary especially).

        returns: list
            [indices, left_position, right_position, disntances1, distances2]
        """
        assert(isinstance(news_collection, NewsWordsCollection))
        assert(isinstance(news_window_size, int))

        l, r, w_from, w_to = self._get_related_to_window_entities_positions(
            news_window_size,
            self.position.left_entity_index, self.position.right_entity_index,
            news_collection.get_words_per_news(self.position.news_ID))

        assert((l > 0) and (r + w_from < w_to))

        x_indices = news_collection.get_embedding_indices_in_window(
            self.position.news_ID,
            total_words_count,
            w_from,
            w_to)

        d1 = self._dist(l, news_window_size)
        d2 = self._dist(r, news_window_size)

        return [x_indices, l, r, d1, d2]

    @staticmethod
    def _dist(pos, size):
        result = []
        for i in range(size):
            result.append(i-pos if i-pos >= 0 else i-pos+size)
        return result

    @staticmethod
    def _get_related_to_window_entities_positions(window_size, left, right, words_in_news):
        """
        returns: tuple
            related left, and related right relation positions and window
            bounds as [w_from, w_to)
        """
        assert(isinstance(left, int) and isinstance(right, int))
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
