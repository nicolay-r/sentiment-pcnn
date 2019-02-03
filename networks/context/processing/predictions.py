import numpy as np
import pickle


class RelationPredictionResultCollection:
    """
    Collection of RelationPredictionResult's
    """

    def __init__(self, relations_count):
        assert(isinstance(relations_count, int))
        self.relations = [None] * relations_count

    def add(self, relation_id, item):
        assert(isinstance(relation_id, int))
        assert(isinstance(item, RelationPredictionResult))
        self.relations[relation_id] = item

    @property
    def PredictionVectorSize(self):
        return len(self.relations[0].Prediction)

    def get_by_relation_id(self, relation_id):
        return self.relations[relation_id]

    @classmethod
    def load(self, pickle_filepath):
        return pickle.load(open(pickle_filepath, 'rb'))

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    def __len__(self):
        return len(self.relations)

    def __iter__(self):
        for relation in self.relations:
            yield relation


class RelationPredictionResult:
    """
    Keeps neural network output information in vector,
    which is related to a certain relation by it's id
    in extracted relation collection.
    """

    def __init__(self, vector):
        assert(isinstance(vector, np.ndarray))
        self._vector = vector

    @property
    def Prediction(self):
        return self._vector
