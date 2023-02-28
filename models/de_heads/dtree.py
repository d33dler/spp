import time
from abc import ABC
from typing import Sequence, Tuple, Any, List, Dict

import mlflow
import numpy as np
import pandas as pd
from easydict import EasyDict
from hyperopt import hp, STATUS_OK, fmin, tpe
from hyperopt.pyll import scope
from scipy.special import softmax
from pandas import DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from torch import nn

import xgboost as xgb
from xgboost import DMatrix

from models.de_heads.dengine import DecisionEngine
from models.interfaces.arch_module import ARCH
from models.utilities.utils import DataHolder


class DTree(DecisionEngine, ABC):
    """
    Decision Tree decision engine type
    """
    all_features: List[str] = []
    search_space = {}
    _float_int_fix = ['max_depth', 'num_parallel_tree', 'num_boost_round']
    def __init__(self, config: EasyDict):
        super().__init__(config)
        self.fine_tuning = self.config.OPTIMIZE
        self.params: dict = self.config.PARAMETERS
        self.optimizer = None
        self.ranks = None
        self.num_classes = None
        if config.TYPE == "CLASSIFIER":
            self.num_classes = config.PARAMETERS["num_class"]
            self.ranks = max(min(3, self.num_classes - 1), 0)

    def optimize(self, train_X: pd.DataFrame, train_Y: pd.DataFrame):
        """
        Hyperparameter finetuning using hyperopt for ensemble tree (gradient boosting) algorithms
        Uses mlflow for logging & tracking progress & performance.
        :param train_X:
        :type train_X:
        :param train_Y:
        :type train_Y:
        :return:
        :rtype:
        """
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
        y_test = y_test.astype(float)
        y_train = y_train.astype(float)
        train = xgb.DMatrix(data=X_train, label=y_train)
        test = xgb.DMatrix(data=X_test, label=y_test)
        labels = np.unique(y_test)
        mlflow.xgboost.autolog(disable=True, silent=True)

        def train_model(params):
            # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.

            # However, we can log additional information by using an MLFlow tracking context manager

            # Train model and record run time
            start_time = time.time()
            booster = xgb.train(params=params, dtrain=train, evals=[(test, "test")], verbose_eval=False,
                                num_boost_round=int(params['num_boost_round']))
            run_time = time.time() - start_time

            # Record Log loss as primary loss for Hyperopt to minimize
            predictions_test = booster.predict(test, output_margin=True)
            loss = log_loss(y_test, predictions_test, labels=labels)

            return {'status': STATUS_OK, 'loss': loss, 'booster': booster.attributes()}

        with mlflow.start_run(run_name='xgb_loss_threshold'):
            best_params = fmin(
                fn=train_model,
                space=self.search_space,
                algo=tpe.suggest,
                loss_threshold=0.1,
                max_evals=50,
                rstate=np.random.default_rng(666),
            )
        mlflow.xgboost.autolog(disable=True)
        mlflow.end_run()
        [best_params.update({k: int(v)}) for k,v in best_params.items() if k in self._float_int_fix]
        return best_params

    def _create_input(self, matrix: np.ndarray):
        return pd.DataFrame(matrix, columns=self.features[self.BASE_FT] + self.features[self.MISC_FT])

    def load(self, state: dict):
        if isinstance(state, bytearray):
            return
        self._init_model(state['type'], **self.params)
        self.model.load_model(state['model'])
        self.params = state['hp']
        self.features = state['features']
        self.is_fit = True

    def dump(self):
        return {
            "model": self.model.save_raw(),
            "hp": self.params,
            "type": self.config.TYPE,
            "features": self.features
        }

    def _init_model(self, _type: str, **params):
        self.model = xgb.Booster()
        if _type == 'REGRESSOR':
            pass
        elif _type == 'CLASSIFIER':
            self.search_space['num_class'] = self.params['num_class'] = self.num_classes
        else:
            raise NotImplementedError('XGB specified model type not supported')
        print(self.params)

    def forward(self, data: DataFrame):
        """
        Predict
        :return:
        :rtype:
        """
        output = self.model.predict(data=DMatrix(data[self.features[self.ALL_FT]]))
        return output

    @staticmethod
    def normalize(x: np.ndarray, axis=0):
        """Softmax function with x as input vector."""
        return softmax(x, axis=0)

    def _fit_classifier(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        if eval_set is None:
            eval_set = [(xgb.DMatrix(data=x, label=y), 'test')]
        self.model = xgb.train(params=self.params, dtrain=DMatrix(data=x, label=y), evals=eval_set, **kwargs)

        [print(f">{o[0]} : {o[1]}") for o in
         sorted(self.model.get_score(importance_type='gain').items(), key=lambda q: q[1], reverse=True)]
        return self.model

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        self.is_fit = True
        if self.fine_tuning:
            self.params.update(self.optimize(x, y))
            print(">BEST PARAMETERS:", self.params)
        self.features[self.ALL_FT] = [f for f in x.columns if f not in y.columns]
        if self.config.TYPE == 'CLASSIFIER':
            return self._fit_classifier(x, y, eval_set, **kwargs)
        return self._fit_regressor(x, y, eval_set, **kwargs)

    def _fit_regressor(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        raise NotImplementedError

    def feature_engineering(self, matrix: np.ndarray, **kwargs):
        tree_df = self._create_input(matrix)
        base_features_ix = tree_df.index.get_indexer(self.features[self.BASE_FT])

        # MIN MAX MEAN STD
        # tree_df['min'] = tree_df.iloc[:, base_features_ix].min(axis=1)
        tree_df['max'] = tree_df.iloc[:, base_features_ix].max(axis=1)
        tree_df['mean'] = tree_df.iloc[:, base_features_ix].mean(axis=1)
        tree_df['std'] = tree_df.iloc[:, base_features_ix].std(axis=1)

        # RANKS
        cls_vals = tree_df.iloc[:, base_features_ix].to_numpy()
        top_k = np.argpartition(-cls_vals, kth=self.ranks, axis=1)[:, :self.ranks]
        tree_df[self.features[self.RANK_FT]] = cls_vals[np.arange(cls_vals.shape[0])[:, None], top_k]
        return tree_df
