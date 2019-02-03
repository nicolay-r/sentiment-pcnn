import io_utils


class DataType:
    """
    Describes collection types that supportes in
    current implementation, and provides by collections.
    """
    Train = u"train"
    Test = u"test"


class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    def __init__(self, model_name):
        assert(isinstance(model_name, unicode))
        self.model_name = model_name

    @staticmethod
    def get_synonyms_collection_filepath():
        return io_utils.get_synonyms_filepath()

    @staticmethod
    def get_files_to_compare_list(method_name, indices):
        return io_utils.create_files_to_compare_list(method_name, indices=indices)

    @staticmethod
    def get_opinion_output_filepath(article_index, model_root):
        return io_utils.get_opin_filepath(article_index, is_etalon=False, root=model_root)

    def get_model_root(self, data_type):
        method_name = None
        if data_type == DataType.Test:
            method_name = self.model_name
        if data_type == DataType.Train:
            method_name = u'{}_train'.format(self.model_name)

        return io_utils.get_method_root(method_name)

    def get_data_indices(self, data_type):
        if data_type == DataType.Test:
            return io_utils.test_indices()
        if data_type == DataType.Train:
            return io_utils.train_indices()
