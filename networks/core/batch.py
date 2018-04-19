import numpy as np
from core.evaluation.labels import Label, PositiveLabel, NegativeLabel, NeutralLabel

from words import NewsWordsCollection
from utils import TextPosition


class MiniBatch:

    def __init__(self, bags):
        assert(isinstance(bags, list))
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
