import numpy as np
from collections import OrderedDict

from core.evaluation.labels import Label

from terms import NewsTermsCollection
from indices import EntityIndices
from extracted_relations import ExtractedRelation
from networks.context.configurations.base import CommonModelSettings
import utils


class Sample(object):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    I_X_INDS = "x_indices"
    I_LABELS = "y"
    I_SUBJ_IND = "subj_inds"
    I_OBJ_IND = "obj_inds"
    I_SUBJ_DISTS = "subj_dist"
    I_OBJ_DISTS = "obj_dist"
    I_POS_INDS = "pos_inds"
    I_NLP_FEATURES = "nlp_features"

    def __init__(self, X, y,
                 subj_ind,
                 obj_ind,
                 dist_from_subj,
                 dist_from_obj,
                 pos_indices,
                 nlp_vector,
                 relation_id):
        """
            X: np.ndarray
                x indices for embedding
            y: int
                uint label
            subj_ind: int
                subject index positions
            obj_ind: int
                object index positions
            dist_from_subj: np.ndarray
            dist_from_obj: np.ndarray
            pos_indices: np.ndarray
            nlp_vector: np.ndarray or None
        """
        assert(isinstance(X, np.ndarray))
        assert(isinstance(y, int))
        assert(isinstance(subj_ind, int))
        assert(isinstance(obj_ind, int))
        assert(isinstance(dist_from_subj, np.ndarray))
        assert(isinstance(dist_from_obj, np.ndarray))
        assert(isinstance(pos_indices, np.ndarray))
        assert(isinstance(nlp_vector, np.ndarray) or nlp_vector is None)
        assert(isinstance(relation_id, int))

        self.relation_id = relation_id

        self.values = OrderedDict(
            [(Sample.I_X_INDS, X),
             (Sample.I_LABELS, y),
             (Sample.I_SUBJ_IND, subj_ind),
             (Sample.I_OBJ_IND, obj_ind),
             (Sample.I_SUBJ_DISTS, dist_from_subj),
             (Sample.I_OBJ_DISTS, dist_from_obj),
             (Sample.I_POS_INDS, pos_indices),
             (Sample.I_NLP_FEATURES, nlp_vector)])

    @property
    def RelationID(self):
        return self.relation_id

    @classmethod
    def from_relation(cls,
                      relation,
                      entity_indices,
                      news_terms_collection,
                      settings):
        assert(isinstance(relation, ExtractedRelation))
        assert(isinstance(entity_indices, EntityIndices))
        assert(isinstance(news_terms_collection, NewsTermsCollection))
        assert(isinstance(settings, CommonModelSettings))

        subj_ind, obj_ind, w_from, w_to = Sample._get_related_to_window_entities_positions(
            settings.TermsPerContext,
            relation.text_position.left_entity_position.TermIndex,
            relation.text_position.right_entity_position.TermIndex)

        pos_indices = utils.calculate_pos_indices_for_terms(
            terms=news_terms_collection.iter_news_terms(relation.text_position.news_ID),
            pos_tagger=settings.PosTagger,
            pad_size=settings.TermsPerContext,
            t_from=w_from,
            t_to=w_to)

        assert((subj_ind > 0) and (obj_ind + w_from < w_to))

        x_indices = utils.calculate_embedding_indices_for_terms(
            terms=news_terms_collection.iter_news_terms(relation.text_position.news_ID),
            term_embedding_matrix=settings.TermEmbeddingMatrix,
            word_embedding=settings.WordEmbedding,
            entity_indices=entity_indices,
            pad_size=settings.TermsPerContext,
            t_from=w_from,
            t_to=w_to)

        dist_from_subj = Sample._dist(subj_ind, settings.TermsPerContext)
        dist_from_obj = Sample._dist(obj_ind, settings.TermsPerContext)

        if relation.opinion_vector is not None:
            nlp_vector = np.array(utils.normalize(relation.opinion_vector.vector))
        else:
            nlp_vector = None

        return cls(X=x_indices,
                   y=relation.label.to_uint(),
                   subj_ind=subj_ind,
                   obj_ind=obj_ind,
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   pos_indices=pos_indices,
                   nlp_vector=nlp_vector,
                   relation_id=relation.relation_id)

    @staticmethod
    def _dist(pos, size):
        result = np.zeros(size)
        for i in range(len(result)):
            result[i] = i-pos if i-pos >= 0 else i-pos+size
        return result

    @staticmethod
    def _get_related_to_window_entities_positions(window_size, left, right):
        """
        returns: tuple
            related left, and related right relation positions and window
            bounds as [w_from, w_to)
        """
        assert(isinstance(window_size, int) and window_size > 0)
        assert(isinstance(left, int) and isinstance(right, int))
        assert(left != right)
        assert(abs(left - right) <= window_size)
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

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def __iter__(self):
        for key, value in self.values.iteritems():
            yield key, value
