import pickle

from core.source.entity import EntityCollection, Entity
from core.processing.lemmatization.base import Stemmer
from networks.context.debug import DebugKeys


class EntityIndices:
    """
    Collects all entities from multiple entities_collections
    """

    def __init__(self, e_value, e_index, stemmer):
        """
        e_value:
            provides lemma by index
        e_index:
            provides index by key
        """
        assert(isinstance(e_value, dict))
        assert(isinstance(e_index, dict))
        assert(isinstance(stemmer, Stemmer))
        self.e_value = e_value
        self.e_index = e_index
        self.stemmer = stemmer

    @classmethod
    def from_entities_collections(cls, entities_collections, stemmer):
        assert(isinstance(stemmer, Stemmer))

        entity_index = 0
        e_value = {}
        e_index = {}
        for entities in entities_collections:
            assert(isinstance(entities, EntityCollection))

            for entity in entities:
                assert(isinstance(entity, Entity))

                lemmas = EntityIndices._value_to_lemmas(entity.value, stemmer)
                key = EntityIndices._lemmas_to_key(lemmas)

                if key not in e_index:
                    e_index[key] = entity_index
                    e_value[entity_index] = entity.value
                    entity_index += 1

        if DebugKeys.EntityIndicesExtractedInfo:
            print "Entities extracted (indices composed): {}".format(len(e_index))

        return cls(e_value, e_index, stemmer)

    @classmethod
    def create_empty(cls, stemmer):
        return cls({}, {}, stemmer)

    @staticmethod
    def _lemmas_to_key(lemmas):
        return '_'.join(lemmas)

    @staticmethod
    def _value_to_lemmas(entity_value, stemmer):
        return stemmer.lemmatize_to_list(entity_value)

    def get_entity_index(self, entity):
        """
        returns: int or None
            int -- index according to the collection this collection,
            None -- in case of undefined 'entity'.
        """
        assert(isinstance(entity, Entity))
        lemmas = self._value_to_lemmas(entity.value, self.stemmer)
        key = self._lemmas_to_key(lemmas)
        if key in self.e_index:
            return self.e_index[key]
        return None

    def value_to_lemmas(self, entity_value):
        assert(isinstance(entity_value, unicode))
        return self._value_to_lemmas(entity_value, self.stemmer)

    def iter_entity_index_and_values(self):
        for e_index, e_value in self.e_value.items():
            yield (e_index, e_value)

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath):
        pickle.load(self, open(pickle_filepath, 'wb'))

    def __len__(self):
        return len(self.e_index)


