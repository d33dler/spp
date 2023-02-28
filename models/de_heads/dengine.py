from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any

import numpy as np
from pandas import DataFrame

from models.interfaces.arch_module import ARCH


class DecisionEngine(ARCH.Child):

    def __init__(self, config):
        super(DecisionEngine, self).__init__(config)
        self._enabled = False
        self._is_fit = False
        self.model: Any = None

    @abstractmethod
    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: DataFrame):
        raise NotImplementedError

    @abstractmethod
    def plot_self(self):
        pass

    @abstractmethod
    def optimize(self, train_X, train_Y):
        raise NotImplementedError

    @property
    def is_fit(self):
        return self._is_fit

    @is_fit.setter
    def is_fit(self, val: bool):
        self._is_fit = val

    @property
    def enabled(self):
        return self._enabled and self._is_fit

    def _create_input(self, matrix: np.ndarray):
        raise NotImplementedError

    def feature_engineering(self, matrix: np.ndarray):

        raise NotImplementedError

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
