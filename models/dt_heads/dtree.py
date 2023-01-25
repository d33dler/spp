from typing import Sequence, Tuple, Any, List

import numpy as np
from scipy.special import softmax
from pandas import DataFrame
from torch import nn

from models.utilities.utils import DataHolder


class DTree(nn.Module):
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

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        raise NotImplementedError

    def create_input(self, matrix: np.ndarray):
        raise NotImplementedError

    def feature_engineering(self, matrix: np.ndarray):
        raise NotImplementedError

    def forward(self, data: DataHolder):
        raise NotImplementedError

    def plot_tree(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    @staticmethod
    def normalize(x: np.ndarray, axis=0):
        """Softmax function with x as input vector."""
        return softmax(x, axis=0)

    def set_fit(self):
        self.is_fit = True
