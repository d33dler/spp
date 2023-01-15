from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
            input = [dt_head.normalize(r) for r in out.detach().cpu().numpy()]
            self.data.X = dt_head.create_input(input)
            return torch.from_numpy(dt_head.forward(self.data)).cuda()
        return out
