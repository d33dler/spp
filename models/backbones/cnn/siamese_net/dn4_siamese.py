from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim

from models.backbones.base import BaseBackbone2d
from models.utilities.custom_loss import NPlusOneTupletLoss
from models.utilities.utils import DataHolder, get_norm_layer, init_weights_kaiming


##############################################################################
# Class: Siamese_ResNet18
##############################################################################

# Model: Siamese_FourLayer_64F
# Input: Query set, Support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class SiameseNetwork(BaseBackbone2d):
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
        self.output_shape = 64
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
            nn.Linear(64 * 21 * 21, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 1024),
        )

        # freeze batchnorm layers
        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]  # , (self.fc, [1, 4])]
        self.lr = model_cfg.LEARNING_RATE
        self.features.apply(init_weights_kaiming)
        self.fc.apply(init_weights_kaiming)
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=0.0001)
        self.criterion = NPlusOneTupletLoss().cuda()

    def forward(self):
        data = self.data
        queries = data.snx_queries
        data.snx_query_f = self.fc(self.features(queries))
        if data.is_training():
            data.snx_positive_f = self.fc(self.features(self.data.snx_positives))
            # construct negatives out of positives for each class (N=50) so negatives = N-1
            negatives = []
            positives = data.snx_positives
            for i in range(len(positives)):
                negatives.append(positives[torch.arange(len(positives)) != i])
            self.data.snx_negative_f = torch.stack(negatives)
            print(queries.shape, positives.shape, self.data.snx_negative_f.shape)
            data.sim_list = self._calculate_cos_similarity(queries, positives, negatives)
            return None
        else:

            support_sets = data.snx_support_sets
            data.snx_support_set_f = self.fc(self.features(support_sets))
            data.sim_list = self._calculate_cos_similarity_support(queries, support_sets)
        return data.sim_list

    def _calculate_cos_similarity(self, queries, positives, negatives):
        sim_list = None

        return sim_list

    def _calculate_cos_similarity_support(self, queries, support_sets):
        sim_list = None
        return sim_list

    def backward(self, *args, **kwargs):
        queries = self.data.snx_query_f
        positives = self.data.snx_positive_f
        negatives = self.data.snx_negative_f
        self.loss = self.criterion(queries, positives, negatives)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def adjust_learning_rate(self, epoch):
        self.scheduler.step()
