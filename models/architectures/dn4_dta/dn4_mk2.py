from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.decomposition import PCA
from torch import nn
from torch.nn.functional import one_hot

from dataset.datasets_csv import CSVLoader
from models.architectures.classifier import ClassifierModel
from models.dt_heads.dtree import DTree


class DN4_DTA(ClassifierModel):
    """
    DN4 MK2 Model

    Structure:
    Deep learning module => K-NN module => Decision Tree
    """

    def __init__(self):
        model_cfg = self.load_config(Path(__file__).parent / 'config.yaml')
        super().__init__(model_cfg)
        self.build()

    def forward(self):
        self.backbone2d(self.data)
        out = self.knn_head(self.data)

        dt_head: DTree = self.dt_head
        if dt_head.is_fit:
            _input = [dt_head.normalize(r) for r in out.detach().cpu().numpy()]  # TODO experiment without normalization
            self.data.X = self.feature_engine(dt_head.create_input(_input), dt_head.base_features, dt_head.deep_features)
            return torch.from_numpy(dt_head.forward(self.data)).float().cuda()
        return out


