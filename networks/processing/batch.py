import numpy as np
from core.evaluation.labels import Label, PositiveLabel, NegativeLabel, NeutralLabel

from networks.processing.samples.base import Sample
from networks.processing.samples.nlp_sample import NLPSample
from words import NewsWordsCollection
from utils import ExtractedRelation


# TODO. Implement additional NLPMiniBatch
class MiniBatch(object):
    """
    Is a neural network batch that is consist of bags.
    """

    def __init__(self, bags):
        assert(isinstance(bags, list))
        self.bags = bags

    def shuffle(self):
        np.random.shuffle(self.bags)

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

        for bag in self.bags:

            assert(isinstance(bag, Bag))
            for sample in bag:

                indices, l, r, d1, d2 = sample.to_network_input(
                    news_collection, news_window_size, total_words_count)

                X.append(indices)
                p1.append(l)
                p2.append(r)
                P1.append(d1)
                P2.append(d2)

                y.append(bag.label.to_uint())

                # self.debug_show(news_collection, indices, l, r,
                #     total_words_count)

        return (X, p1, p2, P1, P2, y)

    def debug_show(self, news_collection, indices, l, r, total_words_count):
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


class NLPMiniBatch(MiniBatch):

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
        features = []
        y = []

        for bag in self.bags:

            assert(isinstance(bag, Bag))
            for sample in bag:

                indices, l, r, d1, d2, nlp_features = sample.to_network_input(
                    news_collection, news_window_size, total_words_count)

                X.append(indices)
                p1.append(l)
                p2.append(r)
                P1.append(d1)
                P2.append(d2)
                features.append(nlp_features)

                y.append(bag.label.to_uint())

                # self.debug_show(news_collection, indices, l, r,
                #     total_words_count)

        return (X, p1, p2, P1, P2, features, y)


class BagsCollection:

    ST_BASE = "base"
    ST_NLP = "nlp"

    def __init__(self, relations, bag_size=40, sample_type="base"):
        assert(isinstance(bag_size, int) and bag_size > 0)  # relations from relationsCollection
        assert(isinstance(relations, list))  # relations from relationsCollection
        assert(sample_type is self.ST_BASE or
               sample_type is self.ST_NLP)

        self.bags = []
        self.bag_size = bag_size
        self.sample_type = sample_type
        pos_bag_ind = self._add_bag(PositiveLabel())
        neg_bag_ind = self._add_bag(NegativeLabel())
        neu_bag_ind = self._add_bag(NeutralLabel())

        for r in relations:
            assert(isinstance(r, ExtractedRelation))

            self._optional_add_in_bag(pos_bag_ind, r.opinion_vector, r.text_position, r.label, sample_type)
            self._optional_add_in_bag(neg_bag_ind, r.opinion_vector, r.text_position, r.label, sample_type)
            self._optional_add_in_bag(neu_bag_ind, r.opinion_vector, r.text_position, r.label, sample_type)

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

    def _optional_add_in_bag(self, bag_index, opinion_vector, position, label, sample_type):
        bag = self.bags[bag_index]
        if bag.label == label:
            bag.add_sample(self._create_sample(opinion_vector,
                                               position,
                                               sample_type))

    def _create_sample(self, opinion_vector, position, sample_type):
        if sample_type is self.ST_BASE:
            return Sample(position)
        elif sample_type is self.ST_NLP:
            return NLPSample(opinion_vector, position)

    def shuffle(self):
        np.random.shuffle(self.bags)

    def _create_minibatch(self, bags_list):
        if self.sample_type is self.ST_BASE:
            return MiniBatch(bags_list)
        elif self.sample_type is self.ST_NLP:
            return NLPMiniBatch(bags_list)

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
        return [self._create_minibatch(bags.tolist()) for bags in grouped_bags]


# TODO. Move it into bags.py
class Bag:
    """
    Bag is a list of positions grouped by 'label'. So each position of
    relation in dataset has the same 'label'
    """

    def __init__(self, label):
        assert(isinstance(label, Label))
        self.samples = []
        self._label = label

    def add_sample(self, sample):
        assert(isinstance(sample, Sample))
        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for p in self.samples:
            yield p

    @property
    def label(self):
        return self._label
