from base import CommonModelSettings


class CNNConfig(CommonModelSettings):

    _terms_per_context_grid = [50, 100, 150, 200]
    _filters_count_grid = [300]

    _window_size = 3
    _filters_count = _filters_count_grid[0]
    _use_bernoulli_mask = False
    _use_nlp_vector = False
    _hidden_size = 300
    _nlp_vector_size = 54

    def iterate_over_grid(self):
        for terms_per_context in self._terms_per_context_grid:
            self._terms_per_context = terms_per_context
            for filter_size in self._filters_count_grid:
                self._filters_count = filter_size
                yield

    @property
    def NLPVectorSize(self):
        return self._nlp_vector_size

    @property
    def UseNLPVector(self):
        return self._use_nlp_vector

    @property
    def UseBernoulliMask(self):
        return self._use_bernoulli_mask

    @property
    def WindowSize(self):
        return self._window_size

    @property
    def FiltersCount(self):
        return self._filters_count

    @property
    def HiddenSize(self):
        return self._hidden_size

    def _internal_get_parameters(self):
        parameters = super(CNNConfig, self)._internal_get_parameters()

        parameters += [
            ("filters_count", self.FiltersCount),
            ("window_size", self.WindowSize),
            ("hidden_size", self.HiddenSize),
            ("use_bernoulli_mask", self.UseBernoulliMask),
            ("use_nlp_vector", self._use_nlp_vector),
            ("nlp_vector_size", self._nlp_vector_size)
        ]

        return parameters
