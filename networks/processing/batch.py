import numpy as np
from core.evaluation.labels import Label, PositiveLabel, NegativeLabel, NeutralLabel

from networks.processing.samples.base import Sample
from words import NewsWordsCollection
from utils import ExtractedRelation


# TODO. Implement additional NLPMiniBatch
class MiniBatch(object):
    """
    Is a neural network batch that is consist of bags.
    """
    I_X_INDS = "x_indices"
    I_LABELS = "y"
    I_SUBJ_IND = "subj_inds"
    I_OBJ_IND = "obj_inds"
    I_SUBJ_DISTS = "subj_dist"
    I_OBJ_DISTS = "obj_dist"
    I_POS_INDS = "pos_inds"
    I_NLP_FEATURES = "nlp_features"

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
        returns: (X, subj_ind, obj_ind, subj_dists, obj_dists, y)

        Note: 'y' has an unsigned labels
        """
        assert(isinstance(news_collection, NewsWordsCollection))

        result = {
            self.I_POS_INDS: [],
            self.I_OBJ_DISTS: [],
            self.I_SUBJ_DISTS: [],
            self.I_OBJ_IND: [],
            self.I_SUBJ_IND: [],
            self.I_X_INDS: [],
            self.I_LABELS: [],
	    self.I_NLP_FEATURES: []
        }

        for bag in self.bags:

            assert(isinstance(bag, Bag))
            for sample in bag:

                assert(isinstance(sample, Sample))

                x_indices, subj_ind, obj_ind, dist_from_subj, dist_from_obj, pos, \
                    nlp_features = sample.to_network_input(news_collection,
                                                           news_window_size,
                                                           total_words_count)

                result[self.I_X_INDS].append(x_indices)
                result[self.I_SUBJ_IND].append(subj_ind)
                result[self.I_OBJ_IND].append(obj_ind)
                result[self.I_SUBJ_DISTS].append(dist_from_subj)
                result[self.I_OBJ_DISTS].append(dist_from_obj)
                result[self.I_POS_INDS].append(pos)
                result[self.I_LABELS].append(bag.label.to_uint())
                result[self.I_NLP_FEATURES].append(nlp_features)

        return result

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


class BagsCollection:

    def __init__(self, relations, bag_size=40):
        assert(isinstance(bag_size, int) and bag_size > 0)  # relations from relationsCollection
        assert(isinstance(relations, list))  # relations from relationsCollection

        self.bags = []
        self.bag_size = bag_size
        pos_bag_ind = self._add_bag(PositiveLabel())
        neg_bag_ind = self._add_bag(NegativeLabel())
        neu_bag_ind = self._add_bag(NeutralLabel())

        for r in relations:
            assert(isinstance(r, ExtractedRelation))

            self._optional_add_in_bag(pos_bag_ind, r.opinion_vector, r.text_position, r.label)
            self._optional_add_in_bag(neg_bag_ind, r.opinion_vector, r.text_position, r.label)
            self._optional_add_in_bag(neu_bag_ind, r.opinion_vector, r.text_position, r.label)

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

    def _optional_add_in_bag(self, bag_index, opinion_vector, position, label):
        bag = self.bags[bag_index]
        if bag.label == label:
            bag.add_sample(self._create_sample(opinion_vector, position))

    def _create_sample(self, opinion_vector, position):
        return Sample(position, opinion_vector)

    def shuffle(self):
        np.random.shuffle(self.bags)

    def _create_minibatch(self, bags_list):
        return MiniBatch(bags_list)

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
