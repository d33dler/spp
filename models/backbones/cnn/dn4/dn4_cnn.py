from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.backbones.base import BaseBackbone2d
from models.clustering.knn import KNN_itc
from models.utilities.utils import DataHolder, get_norm_layer, init_weights_kaiming


##############################################################################
# Class: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class BaselineBackbone2d(BaseBackbone2d):
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
        self.norm_layer = norm_layer
        self.output_channels = 64
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
        self.knn = KNN_itc(data.k_neighbors)

        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]
        self.FREEZE_EPOCH = model_cfg.FREEZE_EPOCH
        self.lr = model_cfg.LEARNING_RATE
        self.features.apply(init_weights_kaiming)
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30, eta_min=0.00005)

    def forward(self):
        # extract features of input1--query image
        data = self.data
        q_embeddings = self.features(data.q_in)
        _, C, _, _ = q_embeddings.size()
        data.q_F = torch.stack(
            [torch.transpose(t.reshape((C, -1)), 0, 1) for t in q_embeddings])

        # get support set embeddings
        L, S, _, _, _ = data.S_in.size()
        S_embeddings = self.features(data.S_in.view(-1, 3, data.S_in.size()[-1], data.S_in.size()[-1]))
        S_embeddings = S_embeddings.view(L, S, C, -1)

        data.S_F = []
        for i in range(L):
            support_set_sam = S_embeddings[i]
            # Reshape to (C, -1)
            support_set_sam = support_set_sam.reshape((C, -1))
            data.S_F.append(support_set_sam)

        av_num = data.get_true_AV() if data.is_training() else 1
        data.sim_list, data.cos_sim = self.knn.forward(data.q_F, data.S_F, av_num, av_num,
                                                       data.cfg.AUGMENTOR.STRATEGY if data.training else None)
        self.data.output = data.sim_list
        return data

    def adjust_learning_rate(self, epoch):
        self.scheduler.step()
