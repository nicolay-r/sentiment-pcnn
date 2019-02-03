import numpy as np
import collections
from random import randint
from core.evaluation.labels import PositiveLabel, NegativeLabel, NeutralLabel
from sample import Sample
from extracted_relations import ExtractedRelation


class BagsCollection:

    def __init__(self,
                 relations,
                 bag_size,
                 create_sample_func,
                 shuffle):
        assert(isinstance(bag_size, int) and bag_size > 0)  # relations from relationsCollection
        assert(isinstance(relations, collections.Iterable))  # relations from relationsCollection
        assert(isinstance(shuffle, bool))

        self.bags = []
        self.bag_size = bag_size

        for relation in relations:
            assert(isinstance(relation, ExtractedRelation))

            if (len(self.bags) == 0) or (len(self.bags[-1]) == bag_size):
                self.bags.append(Bag())

            s = create_sample_func(relation)
            assert(isinstance(s, Sample))

            self.bags[-1].add_sample(s)

        if len(self.bags) > 0:
            self._complete_last_bag(self.bags, bag_size)

        if shuffle:
            np.random.shuffle(self.bags)

    def _complete_last_bag(self, bags, bag_size):
        assert(isinstance(bags, list) and len(bags) > 0)
        assert(isinstance(bag_size, int))

        last_bag = bags[-1]
        assert(isinstance(last_bag, Bag))

        if len(last_bag) == 0:
            return

        while len(last_bag) < bag_size:
            random_bag = bags[randint(0, len(bags)-1)]
            assert(isinstance(random_bag, Bag))
            random_sample_ind = randint(0, len(random_bag._samples)-1)
            last_bag.add_sample(random_bag._samples[random_sample_ind])

    def iter_by_groups(self, bags_per_group):
        """
        returns: list of MiniBatch
        """
        assert(type(bags_per_group) == int and bags_per_group > 0)

        groups_count = len(self.bags) / bags_per_group
        end = 0
        for index in range(groups_count):
            begin = index * bags_per_group
            end = begin + bags_per_group
            yield self.bags[begin:end]

        delta = len(self.bags) - end
        if delta > 0:
            yield self.bags[end:] + self.bags[:bags_per_group - delta]


class Bag:
    """
    Bag is a list of samples
    """

    def __init__(self):
        self._samples = []

    def add_sample(self, sample):
        assert(isinstance(sample, Sample))
        self._samples.append(sample)

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        for sample in self._samples:
            yield sample
