from dataclasses import field

import torch.nn as nn
from easydict import EasyDict
from torch import optim, stack
import torch.nn.functional as F
from models.backbones.base import BaseBackbone2d
from models.clustering.knn import KNN_itc
from models.utilities.utils import DataHolder, init_weights, get_norm_layer


##############################################################################
# Class: Siamese_ResNet18
##############################################################################

# Model: Siamese_FourLayer_64F
# Input: Query set, Support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class Siamese_ResNet18(BaseBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder, config: EasyDict = None):
        super().__init__(self.Config())
        self.data = data
        model_cfg = data.cfg.BACKBONE

        self.require_grad = model_cfg.GRAD

        norm_layer, use_bias = get_norm_layer(model_cfg.NORM)

        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 21 * 21, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1024),
        )
        self.FREEZE_LAYERS = [1, 5, 9, 12]
        self.FREEZE_EPOCH = model_cfg.FREEZE_EPOCH
        self.lr = model_cfg.LEARNING_RATE
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.output_shape = 1024

    def forward(self):
        query = self.data.q_in
        support_sets = self.data.S_in
        query = self.features(query)
        query = query.view(query.size(0), -1)
        query = self.fc(query)

        batch_size = query.size(0)
        distances_list = []

        for support_set in support_sets:
            support = self.features(support_set)
            support = support.view(support.size(0), -1)
            support = self.fc(support)

            distances = F.cosine_similarity(query.unsqueeze(1), support.unsqueeze(0), dim=2)
            distances_list.append(distances)

        distances_tensor = stack(distances_list, dim=1)

        return distances_tensor

