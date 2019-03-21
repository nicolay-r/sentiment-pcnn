#!/usr/bin/python
import numpy as np
from os import path, makedirs
from os.path import dirname
from core.processing.lemmatization.base import Stemmer
from core.evaluation.statistic import FilesToCompare

ignored_entity_values = [u"author", u"unknown"]
TEST = 'test'
TRAIN = 'train'


def train_indices():
    indices = range(1, 46)
    for i in [9, 22, 26]:
        if i in indices:
            indices.remove(i)
    return indices


def test_indices():
    indices = range(46, 76)
    for i in [70]:
        if i in indices:
            indices.remove(i)
    return indices


def collection_indices():
    return train_indices() + test_indices()


def is_ignored_entity_value(entity_value, stemmer):
    """
    entity_value: unicode
    stemmer: Stemmer
    return: bool
    """
    assert(isinstance(entity_value, unicode))
    assert(isinstance(stemmer, Stemmer))
    return stemmer.lemmatize_to_str(entity_value) in ignored_entity_values


def data_root():
    return path.join(dirname(__file__), u"data/")


def eval_root():
    return path.join(data_root(), u"Eval/")


def get_collection_root():
    return path.join(data_root(), u"Collection/")


def get_entity_filepath(index, root=get_collection_root()):
    return path.join(root, u"art{}.ann".format(index))


def get_news_filepath(index, root=get_collection_root()):
    return path.join(root, u"art{}.txt".format(index))


def get_opin_filepath(index, is_etalon, prefix=u'art', root=get_collection_root()):
    assert(isinstance(is_etalon, bool))
    return path.join(root, u"{}{}.opin{}.txt".format(prefix, index, '' if is_etalon else u'.result'))


def get_neutral_filepath(index, is_train, root=get_collection_root()):
    assert(isinstance(is_train, bool))
    return path.join(root, u"art{}.neut.{}.txt".format(index, TRAIN if is_train else TEST))


def get_method_root(method_name):
    result = path.join(eval_root(), method_name)
    if not path.exists(result):
        makedirs(result)
    return result


def get_synonyms_filepath():
    return path.join(data_root(), u"synonyms.txt")


def create_files_to_compare_list(method_name, indices=test_indices(), root=get_collection_root()):
    """
    Create list of comparable opinion files for the certain method.
    method_name: str
    """
    # TODO. Refactor method_name with method_root
    method_root_filepath = get_method_root(method_name)
    return [FilesToCompare(
                get_opin_filepath(i, is_etalon=False, root=method_root_filepath),
                get_opin_filepath(i, is_etalon=True, root=root),
                i) for i in indices]


def indices_to_cv_pairs(cv, indices_list=collection_indices(), shuffle=True, seed=1):
    """
    Splits array of indices into list of pairs (train_indices_list,
    test_indices_list)
    """
    def chunk_it(sequence, num):
        avg = len(sequence) / float(num)
        out = []
        last = 0.0

        while last < len(sequence):
            out.append(sequence[int(last):int(last + avg)])
            last += avg

        return out

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices_list)

    chunks = chunk_it(indices_list, cv)

    for test_index, chunk in enumerate(chunks):
        train_indices = range(len(chunks))
        train_indices.remove(test_index)

        train = [v for train_index in train_indices for v in chunks[train_index]]
        test = chunk

        yield train, test