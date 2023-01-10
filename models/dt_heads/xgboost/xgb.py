from pathlib import Path
from typing import Tuple, Any, Sequence

import numpy as np
import xgboost as xgb
from pandas import DataFrame
import matplotlib.pyplot as plt
from torch import nn

from models.model_utils.utils import load_config


class XGBHead(nn.Module):
    """
    XGBModel head wrapper class providing minimalistic interfacing with the model
    and automatic initialization using the local file config
    """
    model: xgb.XGBModel
    output: np.ndarray = None
    config_id = 'config.yaml'

    def __init__(self):
        super().__init__()
        self.cfg = load_config(Path(Path(__file__).parent / self.config_id))
        t = self.cfg.TYPE
        if t == 'REGRESSOR':
            self.model = xgb.XGBRegressor(self.cfg.PARAMETERS)
        elif t == 'CLASSIFIER':
            self.model = xgb.XGBClassifier(self.cfg.PARAMETERS)
        else:
            raise NotImplementedError('XGB specified model type not supported')
        if self.cfg.LOAD_MODEL:
            self.model.load_model(self.cfg.MODEL_NAME)

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        self.model = self.model.fit(X=x, Y=y, eval_set=eval_set, **kwargs)
        [print(f">{o[0]} : {o[1]}") for o in
         sorted(self.model.get_booster().get_score(importance_type='gain').items(), key=lambda q: q[1], reverse=True)]
        if self.cfg.SAVE_MODEL:
            self.model.save_model(f"{self.cfg.MODEL_NAME}.json")
        return self.model

    def forward(self, X: DataFrame):
        """
        Predict
        :return:
        :rtype:
        """
        self.output = self.model.predict(X=X)
        return self.output

    def plot_importance(self):
        xgb.plot_importance(self.model.get_booster(), importance_type='gain')
        plt.show(block=False)

    def plot_tree(self):
        xgb.plot_tree(self.model.get_booster())
        plt.show(block=False)

    def optimize(self):
        pass  # TODO hyperopt
