#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import io_utils
from classifiers.configuration import CommonSettings
from core.evaluation.labels import NeutralLabel
from core.runtime.relations import Relation
from core.processing.lemmatization.base import Stemmer
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection, Opinion
from core.source.synonyms import SynonymsCollection


def opinions_between_entities(E, diff, news, synonyms, is_ignored_entity, stemmer, sentiment_opins=None):
    """
    Relations that had the same difference
    """
    assert(isinstance(news, News))
    assert(isinstance(synonyms, SynonymsCollection))
    assert(callable(is_ignored_entity))
    assert(isinstance(stemmer, Stemmer))

    def can_add(opinion, neutral_opins):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(neutral_opins, OpinionCollection))

        # Filter if there is a sentiment relation
        if sentiment_opins is not None:
            if sentiment_opins.has_opinion_by_synonyms(opinion):
                return False

        if neutral_opins.has_opinion_by_synonyms(opinion):
            return False

        return True

    def get_entity_synonyms(entity):
        return synonyms.get_synonyms_list(entity.value), \
               synonyms.get_synonym_group_index(entity.value)

    added = set()
    c = OpinionCollection(opinions=None, synonyms=synonyms, stemmer=stemmer)

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):

            if E[i][j] != diff:
                continue

            e1 = news.entities.get_entity_by_index(i)
            e2 = news.entities.get_entity_by_index(j)

            if is_ignored_entity(e1) or is_ignored_entity(e2):
                continue

            if not synonyms.has_synonym(e1.value):
                synonyms.add_synonym(e1.value)

            if not synonyms.has_synonym(e2.value):
                synonyms.add_synonym(e2.value)

            sl1, g1 = get_entity_synonyms(e1)
            sl2, g2 = get_entity_synonyms(e2)

            r_left = sl1[0]
            r_right = sl2[0]

            # Filter the same groups
            if g1 == g2:
                "Entities '{}', and '{}' a part of the same synonym group".format(
                    r_left.encode('utf-8'), r_right.encode('utf-8'))
                continue

            o = c.create_opinion(r_left, r_right, NeutralLabel())
            if can_add(o, c):
                added.add(o.create_value_id())
                c.add_opinion(o)

            o = c.create_opinion(r_right, r_left, NeutralLabel())
            if can_add(o, c):
                added.add(o.create_value_id())
                c.add_opinion(o)

    return c


def make_neutrals(news, synonyms_collection, is_ignored_entity, stemmer, opinions=None):
    assert(isinstance(news, News))
    assert(isinstance(synonyms_collection, SynonymsCollection))
    assert(isinstance(stemmer, Stemmer))
    assert(callable(is_ignored_entity))

    entities = news.entities
    E = np.zeros((entities.count(), entities.count()), dtype='int32')
    for e1 in entities:
        for e2 in entities:
            i = e1.get_int_ID()
            j = e2.get_int_ID()
            relation = Relation(i, j, news)
            E[i-1][j-1] = -1 if i == j else relation.get_distance_in_sentences()

    return opinions_between_entities(E, 0, news, synonyms_collection,
                                     is_ignored_entity,
                                     stemmer=stemmer,
                                     sentiment_opins=opinions)


#
# Main
#
synonyms = SynonymsCollection.from_file(io_utils.get_synonyms_filepath(), stemmer=CommonSettings.stemmer)

#
# Collection as a Train Data
#
for n in io_utils.collection_indices():
    entity_filepath = io_utils.get_entity_filepath(n)
    news_filepath = io_utils.get_news_filepath(n)
    opin_filepath = io_utils.get_opin_filepath(n, is_etalon=True)
    neutral_filepath = io_utils.get_neutral_filepath(n, is_train=True)

    print "Create neutral file: '{}'".format(neutral_filepath)

    entities = EntityCollection.from_file(entity_filepath, stemmer=CommonSettings.stemmer)
    news = News.from_file(news_filepath, entities, stemmer=CommonSettings.stemmer)
    opinions = OpinionCollection.from_file(opin_filepath,
                                           io_utils.get_synonyms_filepath(),
                                           stemmer=CommonSettings.stemmer)

    neutral_opins = make_neutrals(news, synonyms,
                                  lambda e_value: io_utils.is_ignored_entity_value(e_value, CommonSettings.stemmer),
                                  CommonSettings.stemmer, opinions)

    neutral_opins.save(neutral_filepath)

#
# Collection as a Test Data
#
for n in io_utils.collection_indices():
    entity_filepath = io_utils.get_entity_filepath(n)
    news_filepath = io_utils.get_news_filepath(n)
    neutral_filepath = io_utils.get_neutral_filepath(n, is_train=False)

    print "Create neutral file: '{}'".format(neutral_filepath)

    entities = EntityCollection.from_file(entity_filepath, stemmer=CommonSettings.stemmer)
    news = News.from_file(news_filepath, entities, stemmer=CommonSettings.stemmer)

    neutral_opins = make_neutrals(news, synonyms,
                                  lambda e_value: io_utils.is_ignored_entity_value(e_value, CommonSettings.stemmer),
                                  CommonSettings.stemmer)
    neutral_opins.save(neutral_filepath)
