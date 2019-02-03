# -*- coding: utf-8 -*-
from core.source.news import News
from core.source.entity import Entity
from core.source.tokens import Token
from core.runtime.parser import TextParser

from networks.context.debug import DebugKeys


class NewsTermsCollection:

    def __init__(self):
        self.by_id = {}

    def add_news_terms(self, news_terms):
        assert(isinstance(news_terms, NewsTerms))
        assert(news_terms.RelatedNewsID not in self.by_id)
        self.by_id[news_terms.RelatedNewsID] = news_terms

    def iter_news_terms(self, news_ID):
        assert(isinstance(news_ID, int))
        for term in self.by_id[news_ID].iter_terms():
            yield term

    def calculate_min_terms_per_context(self):
        if len(self.by_id) == 0:
            return None

        return min([len(news_terms) for news_terms in self.by_id.itervalues()])


class NewsTerms:
    """
    Extracted News lexemes, such as:
        - news words
        - tokens
        - entities (positions).
    """

    def __init__(self, news_ID, terms, entity_positions):
        assert(isinstance(news_ID, int))
        assert(isinstance(terms, list))
        assert(isinstance(entity_positions, dict))
        self._news_ID = news_ID
        self._terms = terms
        self.entity_positions = entity_positions

        if DebugKeys.NewsTermsStatisticShow:
            self.debug_statistics()
        if DebugKeys.NewsTermsShow:
            self.debug_show()

    @classmethod
    def create_from_news(cls, news_ID, news, keep_tokens):
        assert(isinstance(keep_tokens, bool))
        terms, entity_positions = cls._extract_terms_and_entity_positions(news, keep_tokens)
        return cls(news_ID, terms, entity_positions)

    def iter_terms(self):
        for term in self._terms:
            yield term

    @property
    def RelatedNewsID(self):
        return self._news_ID

    def get_entity_position(self, entity_ID):
        assert(type(entity_ID) == unicode)      # ID which is a part of *.ann files.
        return self.entity_positions[entity_ID]

    @staticmethod
    def _extract_terms_and_entity_positions(news, keep_tokens):
        assert(isinstance(news, News))
        assert(isinstance(keep_tokens, bool))

        terms = []
        entity_positions = {}
        for s_index, s in enumerate(news.sentences):
            s_pos = s.begin
            # TODO: guarantee that entities ordered by e_begin.
            for e_ID, e_begin, e_end in s.entity_info:
                # add terms before entity
                if e_begin > s_pos:
                    parsed_text_before = TextParser.parse(s.text[s_pos:e_begin], keep_tokens=keep_tokens)
                    terms.extend(parsed_text_before.iter_raw_terms())
                # add entity position
                entity_positions[e_ID] = EntityPosition(term_index=len(terms), sentence_index=s_index)
                # add entity_text
                terms.append(news.entities.get_entity_by_id(e_ID))
                s_pos = e_end

            # add text part after last entity of sentence.
            parsed_text_last = TextParser.parse((s.text[s_pos:s.end]), keep_tokens=keep_tokens)
            terms.extend(parsed_text_last.iter_raw_terms())

        return terms, entity_positions

    def debug_show(self):
        for term in self._terms:
            if isinstance(term, unicode):
                print "Word:    '{}'".format(term.encode('utf-8'))
            elif isinstance(term, Token):
                print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                    term.get_original_value().encode('utf-8'))
            elif isinstance(term, Entity):
                print "Entity:  '{}'".format(term.value.encode('utf-8'))
            else:
                raise Exception("unsuported type {}".format(term))

    def debug_statistics(self):
        words = filter(lambda term: isinstance(term, unicode), self._terms)
        tokens = filter(lambda term: isinstance(term, Token), self._terms)
        entities = filter(lambda term: isinstance(term, Entity), self._terms)

        total = len(words) + len(tokens) + len(entities)

        print "Extracted news_words info, NEWS_ID: {}".format(self._news_ID)
        print "\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total)
        print "\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total)
        print "\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total)

    def __len__(self):
        return len(self._terms)


class EntityPosition:

    def __init__(self, term_index, sentence_index):
        self._term_index = term_index
        self._sentence_index = sentence_index

    @property
    def TermIndex(self):
        return self._term_index

    @property
    def SentenceIndex(self):
        return self._sentence_index

