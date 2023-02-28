import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Sequence, List
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.model_selection import train_test_split
from xgboost import DMatrix
from hyperopt import hp, fmin, tpe, STATUS_OK
from hyperopt.pyll import scope
from models.de_heads.dtree import DTree
from models.utilities.utils import load_config, DataHolder


class XGBHead(DTree):
    """
    XGBModel head wrapper class providing minimalistic interfacing with the model
    and automatic initialization using the local file config
    """

    output: np.ndarray = None

    config_id = 'config.yaml'

    def __init__(self, config):
        """
        Initialize model and create search space for parameter fine-tuning using hyperopt
        :param config:
        :type config:
        """
        super().__init__(config)
        self.search_space = {
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.01),
            'num_boost_round': scope.int(hp.quniform('num_boost_round', 50, 200, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 9, 1)),
            'eta': hp.quniform('eta', 0.01, 0.2, 0.02),
            'min_child_weight': hp.quniform('min_child_weight', 1, 5, 0.5),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'gamma': hp.quniform('gamma', 0.1, 10, 1),
            'reg_lambda': hp.quniform('reg_lambda', 0.1, 10, 1),
            'objective': self.params['objective'],
            'eval_metric': self.params['eval_metric'],
            'seed': 123
        }
        [self.search_space.update({k: v}) for k, v in self.params.items() if k not in self.search_space.keys()]
        self._init_model(_type=self.config.TYPE, **self.params)

    @staticmethod
    def get_config():
        return load_config(Path(Path(__file__).parent / "config.yaml"))

    def plot_importance(self):
        xgb.plot_importance(self.model, importance_type='gain')
        plt.show(block=False)

    def plot_self(self):
        xgb.plot_tree(self.model)
        plt.show()
