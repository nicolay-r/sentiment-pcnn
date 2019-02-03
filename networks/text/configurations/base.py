import tensorflow as tf
from core.processing.lemmatization.mystem import MystemWrapper


class MergingSameSentenceMode:

    NoMerge = u"no_merge"
    Average = u"average"


class TextModelSettings(object):

    GPUMemoryFraction = 0.15

    _test_on_epochs = range(1, 10, 1)
    _classes_count = 3
    _batch_size = 2
    _group_size = 5
    _embedding_shape = None
    _default_stemmer = MystemWrapper()
    _dropout = 0.5
    _class_weights = None
    _epoch_to_use = None
    _use_class_weights = True
    _use_sentence_indices = True
    _text_parts_count = 10
    _merging_mode = MergingSameSentenceMode.NoMerge
    _optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=0.1,
        epsilon=10e-6,
        rho=0.95)

    @property
    def Epochs(self):
        return max(self._test_on_epochs)

    @property
    def BatchSize(self):
        return self._batch_size

    @property
    def MergingMode(self):
        return self._merging_mode

    @property
    def GroupSize(self):
        return self._group_size

    @property
    def Stemmer(self):
        return self._default_stemmer

    @property
    def Optimiser(self):
        return self._optimiser

    @property
    def ClassesCount(self):
        return self._classes_count

    @property
    def EmbeddingShape(self):
        return self._embedding_shape

    @property
    def Dropout(self):
        return self._dropout

    @property
    def ClassWeights(self):
        return self._class_weights

    @property
    def UseClassWeights(self):
        return self._use_class_weights

    @property
    def UseSentenceIndices(self):
        return self._use_sentence_indices

    @property
    def EpochToUse(self):
        return self._epoch_to_use

    @property
    def TextPartsCount(self):
        return self._text_parts_count

    @property
    def TestOnEpochs(self):
        return self._test_on_epochs

    def set_class_weights(self, class_weights):
        assert(isinstance(class_weights, list))
        assert(len(class_weights) == self._classes_count)
        self._class_weights = class_weights

    def set_embedding_shape(self, value):
        assert(isinstance(value, tuple) and len(value) == 2)
        self._embedding_shape = value

    def _internal_get_parameters(self):
        return [
            ("use_class_weights", self.UseClassWeights),
            ("dropout", self.Dropout),
            ("classes_count", self.ClassesCount),
            ("class_weights", self.ClassWeights),
            ("default_stemmer",  self.Stemmer),
            ("batch_size", self.BatchSize),
            ("epoch_to_use", self.EpochToUse),
            ("optimizer", self.Optimiser)
        ]

    def get_parameters(self):
        return [list(p) for p in zip(*self._internal_get_parameters())]
