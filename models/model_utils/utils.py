"""
General utilities
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import yaml
from easydict import EasyDict
from pandas import DataFrame
from torch import Tensor


def load_config(config: Path):
    with open(config, mode="r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return EasyDict(cfg)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


@dataclass
class DataHolder:
    """
    Module IO object
    """

    # Input
    q_in: Tensor
    S_in: list
    # Backbone2d-out
    q: Tensor  # TODO identify types
    S: list
    # KNN-out
    knn_list: List[List]
    q_N: Tensor
    # Tree fit input
    X: DataFrame
    y: DataFrame
    eval_set: Tuple[Any, Any]

    # Tree-out
    tree_pred: np.ndarray

    def __init__(self, cfg):
        self.eval_set = None
        self.module_list: List = []
        self.cfg = cfg
