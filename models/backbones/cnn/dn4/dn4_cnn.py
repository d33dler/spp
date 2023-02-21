from dataclasses import field

import torch.nn as nn
from torch import optim

from models.backbones.base import BaseBackbone2d
from models.clustering.knn import KNN_itc
from models.utilities.utils import DataHolder, init_weights, get_norm_layer


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

    #  norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3
    def __init__(self, data: DataHolder):
        super().__init__(self.Config())
        # self.build()
        # super(FourLayer_64F, self).__init__()
        self.data = data
        model_cfg = data.cfg.BACKBONE_2D
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
        self.norm_layers = [1, 5, 9, 12]
        self.lr = model_cfg.LEARNING_RATE
        self.criterion = nn.CrossEntropyLoss().cuda()
        init_weights(self, model_cfg.INIT_WEIGHTS)
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.output_shape = 64
        self.knn = KNN_itc(data.k_neighbors)

    def forward(self):
        # extract features of input1--query image
        data = self.data
        data.q = self.features(data.q_in)
        data.S, data.S_raw = [], []

        # extract features of input2--support set
        for i in range(len(data.S_in)):
            support_set_sam = self.features(data.S_in[i])
            data.S_raw.append(support_set_sam.clone())
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().reshape((C, -1))
            data.S.append(support_set_sam)
        data.sim_list_BACKBONE2D = self.knn.forward(data.q, data.S)
        if self.training:
            self.backward(data.sim_list_BACKBONE2D, data.targets)
        return data


# def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',
#                   use_gpu=True, **kwargs):
#     norm_layer, _ = get_norm_layer(norm_type=norm)
#
#     if use_gpu:
#         assert (torch.cuda.is_available())
#
#     if which_model == 'Conv64F':
#         DN4Net = FourLayer_64F()
#     elif which_model == 'ResNet256F':
#         net_opt = {
#             'userelu': False, 'in_planes': 3, 'dropout': 0.5,
#             'norm_layer': norm_layer
#         }
#         DN4Net = ResNetLike()
#     else:
#         raise NotImplementedError('Model name [%s] is not recognized' % which_model)
#     init_weights(DN4Net, init_type=init_type)
#
#     if use_gpu:
#         DN4Net.cuda()
#
#     if pretrained:
#         DN4Net.load_state_dict(model_root)
#
#     return DN4Net
