from common import CommonSettings

class CNNConfig:

    window_size = 3
    bag_size = 1
    position_size = 1
    bags_per_minibatch = 50
    dropout = 0.5
    words_per_news_grid = [50, 100, 150, 200]
    filter_size_grid = [200, 300]

    filter_size = None
    words_per_news = None
    embedding_shape = None

    test_on_epochs = CommonSettings.test_on_epochs
    optimiser = CommonSettings.optimiser

    def iterate_over_grid(self):
        for words_per_news in self.words_per_news_grid:
            self.words_per_news = words_per_news
            for filter_size in self.filter_size_grid:
                self.filter_size = filter_size
                yield

    @property
    def Epochs(self):
        return max(self.test_on_epochs)
