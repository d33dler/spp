from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys
from yamldataclassconfig import YamlDataClassConfig
from dataclasses import dataclass, field
from models.backbones.base import BaseBackbone2d
from models.backbones.resnet.resnet_256 import ResNetLike
from models.classifiers.knn import ImgtoClass_Metric
from models.model_utils.knn_utils import get_norm_layer, init_weights


##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class FourLayer_64F(BaseBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)
        # LAYER_PADDINGS: list  # [1]
        # LAYER_STRIDES: list  # [1]
        # NUM_FILTERS: list  # [64]
        # LAYER_NUMS: list  # [4]

    #  norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3
    def __init__(self):
        super(FourLayer_64F, self).__init__(self.Config())
        # self.build()

        # super(FourLayer_64F, self).__init__()
        norm_layer = self.cfg.NORMALIZATION_F
        neighbor_k = self.cfg.NUM_CLASSES
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

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

        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes

    def forward(self, input1, input2):
        # extract features of input1--query image
        q = self.features(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            support_set_sam = self.features(input2[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().reshape((C, -1))
            S.append(support_set_sam)

        x = self.imgtoclass(q, S)  # get Batch*num_classes

        return x


def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',
                  use_gpu=True, **kwargs):
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'Conv64F':
        DN4Net = FourLayer_64F(norm_layer=norm_layer, **kwargs)
    elif which_model == 'ResNet256F':
        net_opt = {'userelu': False, 'in_planes': 3, 'dropout': 0.5, 'norm_layer': norm_layer}
        DN4Net = ResNetLike(net_opt)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(DN4Net, init_type=init_type)

    if use_gpu:
        DN4Net.cuda()

    if pretrained:
        DN4Net.load_state_dict(model_root)

    return DN4Net
