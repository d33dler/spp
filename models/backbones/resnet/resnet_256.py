from dataclasses import field

import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys

from models.backbones.base import BaseBackbone2d
from models.classifiers.knn import KNN_itc


##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetLike
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->96->128->256


class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL1', nn.Conv2d(nFin, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(BaseBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)
        USE_RELU = True
        DROPOUT = 0
        IN_PLANES = None
        # LAYER_PADDINGS: list  # [1]
        # LAYER_STRIDES: list  # [1]
        # NUM_FILTERS: list  # [64]
        # LAYER_NUMS: list  # [4]

    def __init__(self): #neighbor_k=3
        super(ResNetLike, self).__init__(self.Config())

        self.in_planes = self.cfg.IN_PLANES
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        if type(self.cfg.NORMALIZATION_F) == functools.partial:
            use_bias = self.cfg.NORMALIZATION_F.func == nn.InstanceNorm2d
        else:
            use_bias = self.cfg.NORMALIZATION_F == nn.InstanceNorm2d

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert (type(self.out_planes) == list)
        assert (len(self.out_planes) == self.num_stages)
        num_planes = [self.out_planes[0], ] + self.out_planes
        userelu = self.cfg.USE_RELU
        dropout = self.cfg.DROPOUT

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))

        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock' + str(i), ResBlock(num_planes[i], num_planes[i + 1]))
            if i < self.num_stages - 2:
                self.feat_extractor.add_module('MaxPool' + str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, True))  # get Batch*256*21*21

        self.imgtoclass = KNN_itc(neighbor_k=self.cfg.NUM_CLASSES)  # Batch*num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.feat_extractor(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            support_set_sam = self.feat_extractor(input2[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().reshape((C, -1))
            S.append(support_set_sam)

        # x = self.imgtoclass(q, S)  # get Batch*num_classes

        return q, S # change to custom IO dataclass
