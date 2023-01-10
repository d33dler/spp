from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import nn

from dataset.datasets_csv import CSVLoader
from models import backbones, classifiers, dt_heads


class ClassifierTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset: CSVLoader):
        super().__init__()
        self.model_cfg = model_cfg  # main cfg! (architecture cfg)
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = model_cfg.CLASS_NAMES
        self.module_topology = ['backbone2d', 'knn_head', 'dt_head']
        self.data = DataHolder(model_cfg)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build(self):
        for module_name in self.module_topology:
            module = getattr(self, '_build_%s' % module_name)()
            self.add_module(module_name, module)

    def forward(self):
        raise NotImplementedError

    def _build_backbone2d(self):
        if self.model_cfg.get("BACKBONE_2D", None) is None:
            raise ValueError('Missing specification of backbone to use')
        m = backbones.__all__[self.model_cfg.BACKBONE_2D]()
        self.data.module_list.append(m)
        return m

    def _build_knn_head(self):
        if self.model_cfg.get("KNN", None) is None:
            return None
        m = classifiers.__all__[self.model_cfg.KNN]()
        self.data.module_list.append(m)
        return m

    def _build_dt_head(self):
        if self.model_cfg.get("DT", None) is None:
            return None
        m = dt_heads.__all__[self.model_cfg.DT]()
        self.data.module_list.append(m)
        return m


@dataclass
class DataHolder:
    """
    Module IO object
    """

    # Input
    q_in: np.ndarray
    S_in: np.ndarray
    # Backbone2d-out
    q: np.ndarray  # TODO identify types
    S: np.ndarray
    # KNN-out
    knn_list: List[List]
    # Tree fit input
    X: DataFrame
    y: DataFrame
    # Tree-out
    tree_pred: np.ndarray

    def __init__(self, cfg):
        self.module_list: List = []
        self.cfg = cf
