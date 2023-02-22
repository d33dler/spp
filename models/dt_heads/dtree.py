import time
from typing import Sequence, Tuple, Any, List

import mlflow
import numpy as np
from hyperopt import hp, STATUS_OK, fmin, tpe
from hyperopt.pyll import scope
from scipy.special import softmax
from pandas import DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from torch import nn

import xgboost as xgb
from models.interfaces.arch_module import ArchM
from models.utilities.utils import DataHolder


class DTree(ArchM.Child):
    is_fit = False
    features: List[str] = []
    base_features: List[str] = []
    deep_features: List[str] = []
    ranking_features: List[str] = []
    all_features: List[str] = []
    max_ix: int = 0
    min_ix: int = 0
    std_ix: int = 0
    mean_ix: int = 0
    ranks_ix: int = 0

    def __init__(self, data: DataHolder):
        super().__init__()
        self.set_optimize(False)
        self.optimizer = None
        self.num_classes = data.num_classes

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        raise NotImplementedError

    def create_input(self, matrix: np.ndarray):
        raise NotImplementedError

    def feature_engineering(self, matrix: np.ndarray):
        raise NotImplementedError

    def forward(self, data: DataFrame):
        raise NotImplementedError

    def plot_tree(self):
        raise NotImplementedError

    def optimize(self, train_X, train_Y):
        def_params = self.params
        search_space = {
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.08, 0.01),
            'max_depth': hp.randint('max_depth', 5, 16),
            'eta': hp.quniform('eta', 0.01, 0.2, 0.02),
            'min_child_weight': hp.quniform('min_child_weight', 2, 10, 0.5),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'gamma': hp.quniform('gamma', 1, 20, 1),
            'reg_lambda': scope.int(hp.quniform('reg_lambda', 1, 30, q=5)),
            'reg_alpha': scope.int(hp.quniform('reg_alpha', 0.1, 10, q=0.5)),
            'objective': def_params['objective'],
            'eval_metric': def_params['eval_metric'],
            'seed': 123
        }

        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
        y_test = y_test.astype(float)
        y_train = y_train.astype(float)
        train = xgb.DMatrix(data=X_train, label=y_train, weight=np.linspace(0.4, 1, len(X_train)))
        test = xgb.DMatrix(data=X_test, label=y_test)
        labels = np.unique(y_test)

        def train_model(params):
            # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
            mlflow.xgboost.autolog(disable=True, silent=True)

            # However, we can log additional information by using an MLFlow tracking context manager
            with mlflow.start_run(nested=True):
                # Train model and record run time
                start_time = time.time()
                params['tree_method'] = 'gpu_hist'
                if self.cfg.TYPE == "CLASSIFIER":
                    params['num_class'] = self.params['num_classes']
                booster = xgb.train(params=params, dtrain=train, evals=[(test, "test")],
                                    early_stopping_rounds=20, num_boost_round=600, verbose_eval=False)
                run_time = time.time() - start_time

                # Record Log loss as primary loss for Hyperopt to minimize
                predictions_test = booster.predict(test, output_margin=True)
                # print(np.unique(predictions_test))
                # print(np.unique(y_test))
                mae_score = log_loss(y_test, predictions_test, labels=labels)

                return {'status': STATUS_OK, 'loss': mae_score, 'booster': booster.attributes()}

        with mlflow.start_run(run_name='xgb_loss_threshold'):
            best_params = fmin(
                fn=train_model,
                space=search_space,
                algo=tpe.suggest,  # stop the grid search once we've reached an MAE of 0.92 or higher
                loss_threshold=0.01,
                max_evals=200,
                rstate=np.random.default_rng(666),
            )
        mlflow.xgboost.autolog(disable=True)
        mlflow.end_run()
        return best_params

    @staticmethod
    def normalize(x: np.ndarray, axis=0):
        """Softmax function with x as input vector."""
        return softmax(x, axis=0)

    def set_fit(self):
        self.is_fit = True
