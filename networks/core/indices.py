from core.source.entity import EntityCollection, Entity
from core.env import stemmer


class EntityIndices:
    """
    Collects all entities from multiple entities_collections
    """

    def __init__(self, entities_collections):
        assert(type(entities_collections) == list)
        entity_index = 0
        self.e_value = {}
        self.e_index = {}
        for entities in entities_collections:
            assert(isinstance(entities, EntityCollection))
            for e in entities:
                assert(isinstance(e, Entity))
                terms = self.value_to_terms(e.value)
                k = self._terms_to_key(terms)
                if k not in self.e_index:
                    self.e_index[k] = entity_index
                    self.e_value[entity_index] = e.value
                    entity_index += 1

    @staticmethod
    def value_to_terms(entity_value):
        assert(type(entity_value) == unicode)
        return stemmer.lemmatize_to_list(entity_value)

    @staticmethod
    def _terms_to_key(terms):
        return '_'.join(terms)

    def get_entity_index(self, entity):
        """
        returns: int or None
            index according to the collection this collection, or None in case
            of undefined 'entity'.
        """
        assert(isinstance(entity, Entity))
        terms = self.value_to_terms(entity.value)
        key = self._terms_to_key(terms)
        if key in self.e_index:
            return self.e_index[key]
        return None

    def iter_entity_index_and_values(self):
        for e_index, e_value in self.e_value.items():
            yield (e_index, e_value)

    def __len__(self):
        return len(self.e_index)


