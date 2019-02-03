from os.path import join, dirname
import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from core.processing.lemmatization.mystem import MystemWrapper
from core.processing.pos.mystem_wrap import POSMystemWrapper
from core.source.embeddings.rusvectores import RusvectoresEmbedding

from networks.context.debug import DebugKeys


class LabelCalculationMode:
    FIRST_APPEARED = u'take_first_appeared'
    AVERAGE = u'average'


class CommonModelSettings(object):

    GPUMemoryFraction = 0.25

    # private settings
    _test_on_epoch = range(0, 30000, 50)
    _use_class_weights = True
    _dropout = 0.5
    _classes_count = 3
    _keep_tokens = True
    _default_stemmer = MystemWrapper()
    _default_pos_tagger = POSMystemWrapper(_default_stemmer.mystem)
    _terms_per_context = 50
    _bags_per_minibatch = 6
    _bag_size = 3
    _word_embedding = None
    _optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=0.5,
        epsilon=10e-6,
        rho=0.95)

    _word_embedding_path = "../../../data/w2v/news_rusvectores2.bin.gz" \
        if not DebugKeys.UseDebugEmbeddingPlaceholder \
        else "../../../data/w2v/banki_ru_300_e5.txt.gz"
    _word_embedding_is_binary = True \
        if not DebugKeys.UseDebugEmbeddingPlaceholder \
        else False

    _term_embedding_matrix = None          # Includes embeddings of: words, entities, tokens.
    _class_weights = None
    _use_pos_emb = True
    _pos_emb_size = 5
    _dist_emb_size = 5
    _relations_label_calc_mode = LabelCalculationMode.AVERAGE

    def __init__(self, load_embedding=True):

        if DebugKeys.LoadWordEmbedding:
            print "Loading embedding: {}".format(self._word_embedding_path)

        if load_embedding:
            self._word_embedding = RusvectoresEmbedding.from_file(
                  filepath=join(dirname(__file__), self._word_embedding_path),
                  binary=self._word_embedding_is_binary,
                  stemmer=self.Stemmer,
                  pos_tagger=self.PosTagger)

        if DebugKeys.LoadWordEmbedding:
            print "Embedding has been loaded."

    @property
    def DistanceEmbeddingSize(self):
        return self._dist_emb_size

    @property
    def RelationLabelCalculationMode(self):
        return self._relations_label_calc_mode

    @property
    def TermEmbeddingMatrix(self):
        return self._term_embedding_matrix

    @property
    def TermEmbeddingShape(self):
        return self._term_embedding_matrix.shape

    @property
    def TotalAmountOfTermsInEmbedding(self):
        """
        Returns vocabulary size -- total amount of words/terms,
        for which embedding has been provided
        """
        return self.TermEmbeddingShape(0)

    def set_term_embedding(self, embedding_matrix):
        self._term_embedding_matrix = embedding_matrix

    def set_class_weights(self, class_weights):
        assert(isinstance(class_weights, list))
        assert(len(class_weights) == self._classes_count)
        self._class_weights = class_weights

    def update_terms_per_context(self, min_possible_value):
        assert(isinstance(min_possible_value, int) and min_possible_value > 0)
        self._terms_per_context = min(min_possible_value, self._terms_per_context)

    @property
    def ClassesCount(self):
        return self._classes_count

    @property
    def Stemmer(self):
        return self._default_stemmer

    @property
    def PosTagger(self):
        return self._default_pos_tagger

    @property
    def ClassWeights(self):
        return self._class_weights

    @property
    def Optimiser(self):
        return self._optimiser

    @property
    def TestOnEpochs(self):
        return self._test_on_epoch

    @property
    def BatchSize(self):
        return self.BagSize * self.BagsPerMinibatch

    @property
    def BagSize(self):
        return self._bag_size

    @property
    def BagsPerMinibatch(self):
        return self._bags_per_minibatch

    @property
    def Dropout(self):
        return self._dropout

    @property
    def KeepTokens(self):
        return self._keep_tokens

    @property
    def TermsPerContext(self):
        return self._terms_per_context

    @property
    def UseClassWeights(self):
        return self._use_class_weights

    @property
    def WordEmbedding(self):
        return self._word_embedding

    @property
    def UsePOSEmbedding(self):
        return self._use_pos_emb

    @property
    def PosEmbeddingSize(self):
        return self._pos_emb_size

    @property
    def Epochs(self):
        return max(self.TestOnEpochs) + 1

    def _internal_get_parameters(self):
        return [
            ("use_class_weights", self.UseClassWeights),
            ("dropout", self.Dropout),
            ("classes_count", self.ClassesCount),
            ("keep_tokens", self.KeepTokens),
            ("class_weights", self.ClassWeights),
            ("default_stemmer",  self.Stemmer),
            ("default_pos_tagger", self.PosTagger),
            ("terms_per_context", self.TermsPerContext),
            ("bags_per_minibatch", self.BagsPerMinibatch),
            ("bag_size", self.BagSize),
            ("batch_size", self.BatchSize),
            ("word_embedding_path", self._word_embedding_path),
            ("use_pos_emb", self.UsePOSEmbedding),
            ("pos_emb_size", self.PosEmbeddingSize),
            ("dist_embedding_size", self.DistanceEmbeddingSize),
            ("relations_label_calc_mode", self.RelationLabelCalculationMode),
            ("optimizer", self.Optimiser)
        ]

    def get_parameters(self):
        return [list(p) for p in zip(*self._internal_get_parameters())]
