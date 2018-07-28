from common import CommonSettings

class CNNConfig:

    bag_size = 1
    position_size = 1
    output_classes = 3
    bags_per_minibatch = 50
    dropout = 0.5

    window_size = 3
    filters_count_grid = [200, 300]
    words_per_news_grid = [50, 100, 150, 200]

    filters_count = None
    words_per_news = None
    embedding_shape = None

    use_bernoulli_mask = False

    use_pos = True 
    pos_emb_size = 5

    use_nlp_vector = False
    nlp_vector_size = 54

    test_on_epochs = CommonSettings.test_on_epochs
    optimiser = CommonSettings.optimiser
    embedding = CommonSettings.load_embedding()

    def iterate_over_grid(self):
        for words_per_news in self.words_per_news_grid:
            self.words_per_news = words_per_news
            for filter_size in self.filters_count_grid:
                self.filters_count = filter_size
                yield

    @property
    def Epochs(self):
        return max(self.test_on_epochs)

    def get_paramters(self):
        return {
            "words_per_news": self.words_per_news,
            "bags_per_batch": self.bags_per_minibatch,
            "bag_size": self.bag_size,
            "use_pos": self.use_pos,
            "pos_emb_size": self.pos_emb_size,
            "channels_count": self.filters_count,
            "window_size": self.window_size,
            "dropout": self.dropout
        }

