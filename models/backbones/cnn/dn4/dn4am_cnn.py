from dataclasses import field

import torch

from models.backbones.base2d import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.utilities.utils import DataHolder, weights_init_kaiming

torch.set_printoptions(profile="full")


##############################################################################
# Class: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class DN4_AM(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder):
        super().__init__(data)
        norm_layer = self.norm_layer
        self.output_channels = 64
        # self.features.append(nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #                                    norm_layer(64),
        #                                    nn.LeakyReLU(0.2, True),
        #
        #                                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #                                    norm_layer(64),
        #                                    nn.LeakyReLU(0.2, True)))
        self.features.apply(weights_init_kaiming)
        # self.post_norm = nn.Sequential(norm_layer(64), nn.LeakyReLU(0.2, True))
        # self.attention.apply(weights_init_kaiming)

        # self.knn = I2C_KNN_AM(self.knn.neighbor_k, _attention_func=attention)

        del self.reg  # = CenterLoss(data.num_classes, 64 * 21 * 21, torch.device('cuda'), reg_lambda=0.1, reg_alpha=0.3).cuda()

    def forward(self):
        data = self.data
        x = self.features(data.q_in)
        data.S_F = self.features(data.S_in)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)  # reshape and permute to match the expected input shape
        attn_output = self.attention(x, x, x)  # use the feature map as query, key and value
        data.q_F = attn_output.permute(1, 2, 0).view(b, c, h, w)  # permute and reshape back to original shape

        qav_num, sav_num = (data.get_qAV(), data.get_SAV()) if data.is_training() else (1, 1)
        data.sim_list = self.knn.forward(data.q_F, data.S_F, qav_num, sav_num,
                                         data.cfg.AUGMENTOR.STRATEGY if data.training else None,
                                         data.cfg.SHOT_NUM)
        self.data.output = data.sim_list
        return data
