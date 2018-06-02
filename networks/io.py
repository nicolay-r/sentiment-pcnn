class NetworkIO:
    """
    Represents Input interface for NeuralNetwork models
    Now exploited (treated) as input interface only
    """

    @staticmethod
    def get_entity_filepath(article_index):
        raise Exception("Not Implemented")

    @staticmethod
    def get_news_filepath(article_index):
        raise Exception("Not Implemented")

    @staticmethod
    def get_opinion_input_filepath(article_index):
        raise Exception("Not Implemented")

    @staticmethod
    def get_opinion_output_filepath(article_index, model_root):
        raise Exception("Not Implemented")

    @staticmethod
    def get_neutral_filepath(article_index, is_train_collection):
        raise Exception("Not Implemented")

    @staticmethod
    def get_model_root(method_name):
        raise Exception("Not Implemented")

    @staticmethod
    def get_files_to_compare_list(method_name, indices, is_train_collection):
        raise Exception("Not Implemented")
