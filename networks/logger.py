import pandas as pd
import numpy as np

from core.evaluation.eval import Evaluator
from context.debug import DebugKeys


# TODO: Move this logger into a root of a networks folder in future.
class CSVLogger:
    """
    This class holds necessary columns for csv table.
    Columns devoted to results per certain epoch for training/testing neural networks.
    """

    F1 = 'f1'
    F1Pos = 'f1-p'
    F1Neg = 'f1-n'
    AvgCost = 'avg-cost'
    AvgAcc = 'avg-acc'
    PosPrecision = 'pos-p'
    NegPrecision = 'neg-p'
    PosRecall = 'pos-r'
    NegRecall = 'neg-r'
    F1Train = 'f1-train'
    Epochs = 'epochs'

    def __init__(self, df, test_on_epochs=None):
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(test_on_epochs, list) or test_on_epochs is None)
        self.df = df
        self.row_index = 0
        self.test_on_epochs = test_on_epochs

    @classmethod
    def create(cls, test_on_epochs):
        assert(isinstance(test_on_epochs, list))

        columns = []
        for i in test_on_epochs:
            columns.append(CSVLogger.get_f1_column_name(i))
            columns.append(CSVLogger._get_f1p_column_name(i))
            columns.append(CSVLogger._get_f1n_column_name(i))
            columns.append(CSVLogger._get_pp_column_name(i))
            columns.append(CSVLogger._get_np_column_name(i))
            columns.append(CSVLogger._get_pr_column_name(i))
            columns.append(CSVLogger._get_nr_column_name(i))
            columns.append(CSVLogger._get_f1_train_column_name(i))
            columns.append(CSVLogger.get_avg_cost_column_name(i))
            columns.append(CSVLogger.Epochs)

        return cls(df=pd.DataFrame(columns=columns),
                   test_on_epochs=test_on_epochs)

    @classmethod
    def load(cls, filepath):
        assert(isinstance(filepath, str) or isinstance(filepath, unicode))
        return cls(df=pd.read_csv(filepath))

    @staticmethod
    def col_name(prefix, index):
        return prefix + '_{}'.format(index)

    @staticmethod
    def get_f1_column_name(index):
        return CSVLogger.col_name(CSVLogger.F1, index)

    @staticmethod
    def get_avg_cost_column_name(index):
        return CSVLogger.col_name(CSVLogger.AvgCost, index)

    @staticmethod
    def get_avg_acc_column_name(index):
        return CSVLogger.col_name(CSVLogger.AvgAcc, index)

    @staticmethod
    def _get_f1p_column_name(index):
        return CSVLogger.col_name(CSVLogger.F1Pos, index)

    @staticmethod
    def _get_f1n_column_name(index):
        return CSVLogger.col_name(CSVLogger.F1Neg, index)

    @staticmethod
    def _get_pp_column_name(index):
        return CSVLogger.col_name(CSVLogger.PosPrecision, index)

    @staticmethod
    def _get_np_column_name(index):
        return CSVLogger.col_name(CSVLogger.NegPrecision, index)

    @staticmethod
    def _get_pr_column_name(index):
        return CSVLogger.col_name(CSVLogger.PosRecall, index)

    @staticmethod
    def _get_nr_column_name(index):
        return CSVLogger.col_name(CSVLogger.NegRecall, index)

    @staticmethod
    def _get_f1_train_column_name(index):
        return CSVLogger.col_name(CSVLogger.F1Train, index)

    def add_column_if_not_exists(self, col_name, value_type):
        assert(isinstance(value_type, type))
        if col_name not in self.df:
            s_length = len(self.df[CSVLogger.get_f1_column_name(self.test_on_epochs[0])])
            default_values = np.zeros(s_length) if value_type != str else ["" for i in range(s_length)]
            self.df[col_name] = pd.Series(default_values, index=self.df.index)

    def create_new_row(self):
        """
        Starts new row for filling related results.
        """
        self.row_index += 1
        self.df.loc[self.row_index] = None
        self.write_value(self.Epochs, max(self.test_on_epochs))
        return self.df.loc[self.row_index]

    def _get_row(self, row_index):
        return self.df.loc[row_index]

    def get_current_row(self):
        return self._get_row(self.row_index)

    def get_row_values(self, row_index, col_prefix=None):
        assert(isinstance(row_index, int))

        if col_prefix is None:
            return self._get_row(row_index)

        return [self.df.loc[row_index][self.col_name(col_prefix, e_index)]
                for e_index in range(len(self.df.columns))
                if self.col_name(col_prefix, e_index) in self.df]

    def get_row_value(self, row_index, col_name):
        return self.df.col[row_index][col_name]

    def write_evaluation_results(self, current_epoch, result_test, result_train, avg_cost, avg_acc):
        assert(isinstance(current_epoch, int))
        assert(isinstance(result_test, dict))
        assert(isinstance(result_train, dict))

        i = current_epoch

        self.write_value(self.get_f1_column_name(i), result_test[Evaluator.C_F1],
                         debug=DebugKeys.LoggerEvaluationF1)
        self.write_value(self._get_f1p_column_name(i), result_test[Evaluator.C_F1_POS], True)
        self.write_value(self._get_f1n_column_name(i), result_test[Evaluator.C_F1_NEG], True)
        self.write_value(self._get_pp_column_name(i), result_test[Evaluator.C_POS_PREC], True)
        self.write_value(self._get_np_column_name(i), result_test[Evaluator.C_NEG_PREC], True)
        self.write_value(self._get_pr_column_name(i), result_test[Evaluator.C_POS_RECALL], True)
        self.write_value(self._get_nr_column_name(i), result_test[Evaluator.C_NEG_RECALL], True)
        self.write_value(self._get_f1_train_column_name(i), result_train[Evaluator.C_F1],
                         debug=DebugKeys.LoggerEvaluationF1Train)
        self.write_value(self.get_avg_cost_column_name(i), avg_cost,
                         debug=DebugKeys.LoggerEvaluationAvgCost)
        self.write_value(self.get_avg_acc_column_name(i), avg_acc,
                         debug=DebugKeys.LoggerEvaluationAvgAcc)

        return

    def write_value(self, column_name, value, debug=False):

        if debug:
            print "set value: df[{}]['{}'] = {}".format(self.row_index, column_name, value)

        self.df.loc[self.row_index, column_name] = value
