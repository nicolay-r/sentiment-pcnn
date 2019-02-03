import numpy as np
import collections

from core.processing.pos.mystem_wrap import POSTagger
from core.source.entity import Entity
from core.source.tokens import Tokens, Token
from core.source.embeddings.base import Embedding
from core.source.embeddings.tokens import TokenEmbeddingVectors

from networks.context.debug import DebugKeys

from indices import EntityIndices


def distances_from_position(pos, size):
    """
    Returns vector of distances from position in array of size 'size'
    pos: int
    size: int
    returns: list
    """
    result = []

    for i in range(size):
        result.append(i-pos if i-pos >= 0 else i-pos+size)

    return result


def pad(lst, pad_size, filler):
    """
    Pad list ('lst') with additional elements (filler)

    lst: list
    pad_size: int
        result size
    filler: int
    returns: None
        inplace
    """
    assert(isinstance(lst, list))
    assert(pad_size - len(lst) >= 0)
    lst.extend([filler] * (pad_size - len(lst)))


def calculate_embedding_indices_for_terms(terms,
                                          term_embedding_matrix,
                                          word_embedding,
                                          entity_indices,
                                          pad_size=None,
                                          t_from=None, t_to=None):
    """
    terms: list
        list that includes words, tokens, entities.
    entity_indices: list
    t_from: None or int
        position of initial term, is a beginning of a window in news
    t_to: None or int
        position of last term (not included), is an end of window in news

    returns: list int
        list of indices
    """
    # O(N^2) because of search at embedding by word to obtain related
    # index.
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(term_embedding_matrix, np.ndarray))
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(entity_indices, EntityIndices))
    assert((isinstance(pad_size, int) and pad_size > 0) or pad_size is None)
    assert(t_from is None or isinstance(t_from, int))
    assert(t_to is None or isinstance(t_to, int))

    indices = []
    embedding_offsets = TermsEmbeddingOffsets(words_count=len(word_embedding.vocab),
                                              entities_count=len(entity_indices))

    unknown_word_embedding_index = embedding_offsets.get_token_index(
        TokenEmbeddingVectors.get_token_index(Tokens.UNKNOWN_WORD))

    debug_words_found = 0
    debug_words_count = 0
    for i, term in enumerate(terms):

        if t_from is not None and t_to is not None:
            if i < t_from or i >= t_to:
                continue

        if isinstance(term, unicode):
            index = unknown_word_embedding_index
            words = word_embedding.Stemmer.lemmatize_to_list(term)
            if len(words) > 0:
                word = words[0]
                if word in word_embedding:
                    index = embedding_offsets.get_word_index(word_embedding.find_index_by_word(word))
                debug_words_found += int(word in word_embedding)
                debug_words_count += 1
        elif isinstance(term, Token):
            index = embedding_offsets.get_token_index(TokenEmbeddingVectors.get_token_index(term.get_token_value()))
        elif isinstance(term, Entity):
            index = embedding_offsets.get_entity_index(entity_indices.get_entity_index(term))
        else:
            raise Exception("Unsuported type {}".format(term))

        indices.append(index)

    if pad_size is not None:
        pad(indices, pad_size=pad_size, filler=unknown_word_embedding_index)

    if DebugKeys.EmbeddingIndicesPercentWordsFound:
        print "words found: {} ({}%)".format(debug_words_found, 100.0 * debug_words_found / debug_words_count)
        print "words missed: {} ({}%)".format(debug_words_count - debug_words_found,
                                              100.0 * (debug_words_count - debug_words_found) / debug_words_count)

    return np.array(indices)


def calculate_pos_indices_for_terms(terms, pos_tagger, pad_size=None, t_from=None, t_to=None):
    """
    terms: list
    pad_size: int
    t_from: None or int
        position of initial term, is a beginning of a window in news
    t_to: None or int
        position of last term (not included), is an end of window in news

    returns: list of int
        list of pos indices
    """
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(pos_tagger, POSTagger))
    assert((isinstance(pad_size, int) and pad_size > 0) or pad_size is None)
    assert(t_from is None or isinstance(t_from, int))
    assert(t_to is None or isinstance(t_to, int))

    indices = []

    for index, term in enumerate(terms):

        if t_from is not None and t_to is not None:
            if index < t_from or index >= t_to:
                continue

        if isinstance(term, Token):
            pos = pos_tagger.Empty
        elif isinstance(term, unicode):
            pos = pos_tagger.get_term_pos(term)
        else:
            pos = pos_tagger.Unknown

        indices.append(pos_tagger.pos_to_int(pos))

    if pad_size is not None:
        pad(indices, pad_size=pad_size, filler=pos_tagger.pos_to_int(pos_tagger.Unknown))

    return np.array(indices)


def create_term_embedding_matrix(word_embedding, entity_indices):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - entities embeddings
        - token embeddings

    word_embedding: Embedding
        embedding vocabulary for words
    entity_indices: EntityIndices or None
    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(entity_indices, EntityIndices))

    embedding_offsets = TermsEmbeddingOffsets(words_count=len(word_embedding.vocab),
                                              entities_count=len(entity_indices))
    token_embedding = TokenEmbeddingVectors(word_embedding.VectorSize)
    matrix = np.zeros((embedding_offsets.TotalCount, word_embedding.VectorSize))

    # words.
    for word, info in word_embedding.vocab.items():
        index = info.index
        matrix[embedding_offsets.get_word_index(index)] = word_embedding.get_vector_by_index(index)

    # entities.
    for e_index, e_value in entity_indices.iter_entity_index_and_values():
        terms = entity_indices.value_to_lemmas(e_value)
        matrix[embedding_offsets.get_entity_index(e_index)] = _get_mean_word2vec_vector(word_embedding, terms)

    # tokens.
    for token_value in token_embedding:
        index = token_embedding.get_token_index(token_value)
        matrix[embedding_offsets.get_token_index(index)] = token_embedding[token_value]

    if DebugKeys.DisplayTermEmbeddingParameters:
        print "Term matrix shape: {}".format(matrix.shape)
        embedding_offsets.debug_print()

    return matrix


def _get_mean_word2vec_vector(embedding, terms):
    """
    embedding: Embedding
        embedding vocabulary for words
    terms: list
    """
    mean_vector = np.zeros(embedding.VectorSize, dtype=np.float32)
    for term in terms:
        if term in embedding:
            mean_vector = mean_vector + embedding[term]
    return mean_vector


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class TermsEmbeddingOffsets:
    """
    Describes indices distibution within a further TermsEmbedding.
    """

    def __init__(self, words_count, entities_count, tokens_count=TokenEmbeddingVectors.count()):
        assert(isinstance(words_count, int))
        assert(isinstance(entities_count, int))
        assert(isinstance(tokens_count, int))
        self.words_count = words_count
        self.entities_count = entities_count
        self.tokens_count = tokens_count

    @property
    def TotalCount(self):
        return self.entities_count + self.words_count + self.tokens_count

    def get_word_index(self, index):
        return index

    def get_entity_index(self, index):
        return self.words_count + index

    def get_token_index(self, index):
        return self.words_count + self.entities_count + index

    def debug_print(self):
        print "Term embedding matrix details ..."
        print "\t\tWords count: {}".format(self.words_count)
        print "\t\tEntities count: {}".format(self.entities_count)
        print "\t\tTokens count: {}".format(self.tokens_count)
