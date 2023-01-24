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
    normalizer = torch.nn.BatchNorm2d(64)
    pca_n = PCA(n_components=64)
    def __init__(self):
        model_cfg = self.load_config(Path(__file__).parent / 'config.yaml')
        super().__init__(model_cfg)
        self.build()

    def forward(self):
        self.backbone2d(self.data)
        out = self.knn_head(self.data)

        dt_head: DTree = self.dt_head
        if dt_head.is_fit:
            _input = np.asarray([dt_head.normalize(x) for x in out.detach().cpu().numpy()])
            self.data.X = self.feature_engine(dt_head.create_input(_input), dt_head.base_features, dt_head.deep_features)
            o = torch.from_numpy(dt_head.forward(self.data).astype(np.int64))
            o = one_hot(o, self.num_classes).float().cuda()
            return o
        return out
