import pandas as pd
import numpy as np

from core.evaluation.eval import Evaluator


class PandasResultLogger:
    """
    This class holds necessary columns for csv table.
    Columns devoted to results per certain epoch for training/testing neural networks.
    """

    col_f1 = 'f1_{}'
    col_f1p = 'f1-p_{}'
    col_f1n = 'f1-n_{}'
    col_avg_cost = 'avg-cost_{}'
    col_pp = 'pos-p_{}'
    col_np = 'neg-p_{}'
    col_pr = 'pos-r_{}'
    col_nr = 'neg-r_{}'
    col_f1_train = 'f1-train_{}'

    def __init__(self, test_on_epochs):

        columns = []

        for i in test_on_epochs:
            columns.append(self._get_f1_column_name(i))
            columns.append(self._get_f1p_column_name(i))
            columns.append(self._get_f1n_column_name(i))
            columns.append(self._get_pp_column_name(i))
            columns.append(self._get_np_column_name(i))
            columns.append(self._get_pr_column_name(i))
            columns.append(self._get_nr_column_name(i))
            columns.append(self._get_f1_train_column_name(i))
            columns.append(self._get_avg_cost_column_name(i))

        self.df = pd.DataFrame(columns=columns)

        self.test_on_epochs = test_on_epochs
        self.row_index = 0

    @staticmethod
    def _get_f1_column_name(index):
        return PandasResultLogger.col_f1.format(index)

    @staticmethod
    def _get_avg_cost_column_name(index):
        return PandasResultLogger.col_avg_cost.format(index)

    @staticmethod
    def _get_f1p_column_name(index):
        return PandasResultLogger.col_f1p.format(index)

    @staticmethod
    def _get_f1n_column_name(index):
        return PandasResultLogger.col_f1n.format(index)

    @staticmethod
    def _get_pp_column_name(index):
        return PandasResultLogger.col_pp.format(index)

    @staticmethod
    def _get_np_column_name(index):
        return PandasResultLogger.col_np.format(index)

    @staticmethod
    def _get_pr_column_name(index):
        return PandasResultLogger.col_pr.format(index)

    @staticmethod
    def _get_nr_column_name(index):
        return PandasResultLogger.col_nr.format(index)

    @staticmethod
    def _get_f1_train_column_name(index):
        return PandasResultLogger.col_f1_train.format(index)

    def add_column_if_not_exists(self, col_name, value_type):
        assert(isinstance(value_type, type))
        if col_name not in self.df:
            s_length = len(self.df[PandasResultLogger._get_f1_column_name(self.test_on_epochs[0])])
            default_values = np.zeros(s_length) if value_type != str else ["" for i in range(s_length)]
            self.df[col_name] = pd.Series(default_values, index=self.df.index)

    def create_new_row(self):
        """
        Starts new row for filling related results.
        """
        self.row_index += 1
        self.df.loc[self.row_index] = None
        return self.df.loc[self.row_index]

    def get_current_row(self):
        return self.df.loc[self.row_index]

    def write_evaluation_results(self, current_epoch, result_test, result_train, avg_cost):
        assert(isinstance(current_epoch, int))
        assert(isinstance(result_test, dict))
        assert(isinstance(result_train, dict))

        i = current_epoch

        self.write_value(self._get_f1_column_name(i), result_test[Evaluator.C_F1], debug=True)
        self.write_value(self._get_f1p_column_name(i), result_test[Evaluator.C_F1_POS], debug=True)
        self.write_value(self._get_f1n_column_name(i), result_test[Evaluator.C_F1_NEG], debug=True)
        self.write_value(self._get_pp_column_name(i), result_test[Evaluator.C_POS_PREC], debug=True)
        self.write_value(self._get_np_column_name(i), result_test[Evaluator.C_NEG_PREC], debug=True)
        self.write_value(self._get_pr_column_name(i), result_test[Evaluator.C_POS_RECALL], debug=True)
        self.write_value(self._get_nr_column_name(i), result_test[Evaluator.C_NEG_RECALL], debug=True)
        self.write_value(self._get_f1_train_column_name(i), result_train[Evaluator.C_F1], debug=True)
        self.write_value(self._get_avg_cost_column_name(i), avg_cost, debug=True)

        return

    def write_value(self, column_name, value, debug=False):

        if debug:
            print "set value: df[{}]['{}'] = {}".format(self.row_index, column_name, value)

        self.df.set_value(col=column_name, index=self.row_index, value=value)
