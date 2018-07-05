from networks.callback import Callback
from networks.logger import PandasResultLogger


class PandasLoggerCallback(Callback):

    col_train_relations_count = 'train_relations'
    col_test_relations_count = 'test_relations'
    col_epochs = 'epochs'

    def __init__(self, epochs, test_on_epochs, csv_filepath, model_name):
        """
        epochs: int
           amount of epochs to train
        test_on_epochs: list
            list of epochs at which it is necessary to call test.
        csv_filepath: str
            output filepath
        model_name: str
        """
        self.logger = PandasResultLogger(test_on_epochs)
        self.logger.add_column_if_not_exists(self.col_train_relations_count, int)
        self.logger.add_column_if_not_exists(self.col_test_relations_count, int)
        self.logger.add_column_if_not_exists(self.col_epochs, int)

        self.model = None
        self.current_epoch = 0
        self.epochs = epochs
        self.test_on_epochs = test_on_epochs
        self.csv_filepath = csv_filepath
        self.model_name = model_name

    def on_initialized(self, model):

        self.current_epoch = 0
        row = self.logger.create_new_row()

        row[self.col_train_relations_count] = len(model.train_relations_collection.relations)
        row[self.col_test_relations_count] = len(model.test_relations_collection.relations)
        row[self.col_epochs] = self.epochs

        for key, value in model.network.ParametersDictionary.iteritems():
            col_name = "col_{}".format(key)
            self.logger.add_column_if_not_exists(col_name, type(value))
            self.logger.write_value(col_name, value, debug=False)

        self.model = model

    def on_epoch_finished(self, avg_cost):

        print "avg_cost type: {}".format(type(avg_cost))
        print "avg_cost value: '{}'".format(avg_cost)

        self.current_epoch += 1
        print "{}: Epoch: {}, average cost: {:.3f}".format(
            str(datetime.datetime.now()), self.current_epoch, avg_cost)

        if self.current_epoch not in self.test_on_epochs:
            return

        self.logger.write_evaluation_results(
            current_epoch=self.current_epoch,
            result_test=self.model.predict(model_name=self.model_name, test_collection=True, debug=False),
            result_train=self.model.predict(model_name=self.model_name, test_collection=False, debug=False),
            avg_cost=avg_cost)

        # save
        self.logger.df.to_csv(self.csv_filepath)