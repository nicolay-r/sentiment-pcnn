#!/usr/bin/python
import io
import numpy as np
from os import path, makedirs, mkdir
from os.path import dirname
from gensim.models.word2vec import Word2Vec
from core.processing.lemmatization.base import Stemmer
from core.evaluation.statistic import FilesToCompare

ignored_entity_values = [u"author", u"unknown"]
TEST = 'test'
TRAIN = 'train'


def get_server_root():
    return u'/home/nicolay/storage/disk/homes/nicolay/datasets/news'


def read_prepositions(filepath):
    prepositions = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            prepositions.append(line.strip())

    return prepositions


def read_lss(filepath):
    words = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words.append(line.lower().strip())

    return words


def read_feature_names():
    features = []
    with open(get_feature_names_filepath(), 'r') as f:
        for l in f.readlines():
            features.append(l.strip())
    return features


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


def test_root():
    result = path.join(data_root(), u"Test/")
    if not path.exists(result):
        makedirs(result)
    return result


def pretrained_root():
    result = path.join(data_root(), u"Pretrained/")
    if not path.exists(result):
        makedirs(result)
    return result


def load_w2v_model(filepath=path.join(data_root(), u"w2v/news_rusvectores2.bin.gz"), binary=True):
    print u"Loading word2vec model '{}' ...".format(filepath)
    w2v_model = Word2Vec.load_word2vec_format(filepath, binary=binary)
    return w2v_model


def graph_root():
    result = path.join(data_root(), u"Graphs/")
    if not path.exists(result):
        makedirs(result)
    return result


def get_collection_root():
    return path.join(data_root(), u"Collection/")


def get_entity_filepath(index, root=get_collection_root()):
    return path.join(root, u"art{}.ann".format(index))


def get_news_filepath(index, root=get_collection_root()):
    return path.join(root, u"art{}.txt".format(index))


def get_opin_filepath(index, is_etalon, prefix=u'art', root=get_collection_root()):
    assert(isinstance(is_etalon, bool))
    return path.join(root, u"{}{}.opin{}.txt".format(prefix, index, '' if is_etalon else u'.result'))


def get_constraint_opin_filepath(index, root=get_collection_root()):
    return path.join(root, u"art{}.opin.graph.txt".format(index))


def get_neutral_filepath(index, is_train, root=get_collection_root()):
    assert(isinstance(is_train, bool))
    return path.join(root, u"art{}.neut.{}.txt".format(index, TRAIN if is_train else TEST))


def get_vectors_filepath(index, is_train, root=get_collection_root()):
    assert(isinstance(is_train, bool))
    return path.join(root, u"art{}.vectors.{}.txt".format(index, TRAIN if is_train else TEST))


def eval_rfe_root():
    result = path.join(eval_root(), u"rfe")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_sfm_root():
    result = path.join(eval_root(), u"sfm")
    if not path.exists(result):
        makedirs(result)
    return result

def eval_sfm_cv_root(cv_count):
    assert(isinstance(cv_count, int))
    result = path.join(eval_root(), u"sfm_{}".format(cv_count))
    if not path.exists(result):
        makedirs(result)
    return result


def eval_ec_root():
    """ class elemination root
    """
    result = path.join(eval_root(), u"ce")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_univariate_root():
    result = path.join(eval_root(), u"uv")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_default_root():
    result = path.join(eval_root(), u"default")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_default_cv_root(cv_count):
    assert(isinstance(cv_count, int))
    result = path.join(eval_root(), u"default_cv_{}".format(cv_count))
    if not path.exists(result):
        makedirs(result)
    return result


def eval_ensemble_cv_root(cv_count):
    assert(isinstance(cv_count, int))
    result = path.join(eval_root(), u"ensemble_cv_{}".format(cv_count))
    if not path.exists(result):
        makedirs(result)
    return result


def eval_ensemble_root():
    result = path.join(eval_root(), u"ensemble")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_baseline_root():
    result = path.join(eval_root(), u"baseline")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_baseline_cv_root(cv_count):
    assert(isinstance(cv_count, int))
    result = path.join(eval_root(), u"baseline_cv_{}".format(cv_count))
    if not path.exists(result):
        makedirs(result)
    return result


def eval_features_filepath():
    return path.join(eval_root(), u'features')


def get_method_root(method_name):
    result = path.join(eval_root(), method_name)
    if not path.exists(result):
        makedirs(result)
    return result


def get_vectors_list(is_train, indices=None, root=get_collection_root()):
    if indices is None:
        indices = train_indices() if is_train else test_indices()
    return [get_vectors_filepath(i, is_train, root) for i in indices]


def get_etalon_root():
    return path.join(data_root(), u"Etalon/")


def get_synonyms_filepath():
    return path.join(data_root(), u"synonyms.txt")


def get_feature_names_filepath():
    return path.join(data_root(), u"feature_names.txt")


def get_log_csv_filepath(file_name):
    return path.join(eval_root(), u'{}.csv'.format(file_name))


def save_test_opinions(test_opinions, method_name, indices=test_indices()):
    """
    Save list of opinions
    """
    method_root = get_method_root(method_name)
    if not path.exists(method_root):
        mkdir(method_root)

    for i, test_index in enumerate(indices):
        # TODO. should guarantee that order the same as during reading operation.
        test_opinions[i].save(get_opin_filepath(test_index, is_etalon=False, root=method_root))


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
