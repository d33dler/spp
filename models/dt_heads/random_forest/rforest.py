from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Sequence, List

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame
import matplotlib.pyplot as plt
from xgboost import DMatrix

from models.dt_heads.dtree import DTree
from models.utilities.utils import load_config, DataHolder


class RandomForestHead(DTree):
    """
    XGBModel head wrapper class providing minimalistic interfacing with the model
    and automatic initialization using the local file config
    """
    model: xgb.XGBModel
    output: np.ndarray = None
    config_id = 'config.yaml'

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.features: List[str] = []
        self.cfg = load_config(Path(Path(__file__).parent / self.config_id))
        t = self.cfg.TYPE
        self.params: dict = self.cfg.PARAMETERS
        if t == 'REGRESSOR':
            self.model = xgb.XGBRFRegressor(**self.params)
        elif t == 'CLASSIFIER':
            self.params['num_classes'] = num_classes
            print(self.params)
            self.model = xgb.XGBClassifier(**self.params)
        else:
            raise NotImplementedError('XGB specified model type not supported')
        if self.cfg.LOAD_MODEL:
            self.model.load_model(self.cfg.MODEL_NAME)

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        self.set_fit()
        self.features = [f for f in x.columns if f not in y.columns]
        if self.cfg.TYPE == 'CLASSIFIER':
            return self._fit_classifier(x, y, eval_set, **kwargs)
        return self._fit_regressor(x, y, eval_set, **kwargs)

    def _fit_classifier(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        if eval_set is None:
            eval_set = [(x, y)]
        self.model = self.model.fit(X=x, y=y, eval_set=eval_set)

        [print(f">{o[0]} : {o[1]}") for o in
         sorted(self.model.get_booster().get_score(importance_type='gain').items(), key=lambda q: q[1], reverse=True)]
        if self.cfg.SAVE_MODEL:
            now = datetime.now()
            date = now.strftime('%m_%d_%H_%M_%S')
            self.model.save_model(f"results/{self.cfg.MODEL_NAME}_[{date}].json")
        return self.model

    def _fit_regressor(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        raise NotImplementedError

    def create_input(self, matrix: np.ndarray):
        matrix = self.feature_engineering(matrix)
        return pd.DataFrame(matrix, columns=self.base_features)

    def feature_engineering(self, matrix: np.ndarray):
        return matrix

    def forward(self, data: DataHolder):
        """
        Predict
        :return:
        :rtype:
        """
        output = self.model.predict(X=data.X[self.all_features])

        return output

    def plot_importance(self):
        xgb.plot_importance(self.model.get_booster(), importance_type='gain')
        plt.show(block=False)

    def plot_tree(self):
        xgb.plot_tree(self.model.get_booster())
        plt.show()

    def optimize(self):
        pass  # TODO hyperopt
