from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim, stack, Tensor
import torch.nn.functional as F
from models.backbones.base import BaseBackbone2d
from models.clustering.knn import KNN_itc
from models.utilities.custom_loss import NPairLoss, NPlusOneTupletLoss
from models.utilities.utils import DataHolder, init_weights, get_norm_layer, geometric_mean


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
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
        )

        self.knn = KNN_itc(data.k_neighbors, True)
        # freeze batchnorm layers
        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]  # , (self.fc, [1, 4])]
        self.lr = model_cfg.LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.criterion = NPlusOneTupletLoss().cuda()

    def forward(self):
        query = self.data.q_in
        self.data.q = self.features(query)
        # query = query.view(query.size(0), -1)
        # query = self.fc(query)
        data = self.data
        data.S = []
        for i in range(len(data.S_in)):
            support_set_sam = self.features(data.S_in[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().reshape((C, -1))
            data.S.append(support_set_sam)
        data.sim_list, data.cos_sim = self.knn.forward(data.q, data.S, data.av_num)
        self.data.output = data.sim_list
        B, L, _ = data.cos_sim.shape
        # Assume targets is a 1D tensor of shape (B,)
        targets = data.targets

        positives = []
        negatives = []

        for i in range(B):
            # For each query, get the similarities to the positive class
            positive_similarities: Tensor = data.cos_sim[i, targets[i]]

            # Use the mean or median of the positive similarities as the positive instance
            positive = positive_similarities  # or positive_similarities.mean()

            # Get the similarities to the negative classes
            negative_similarities = torch.cat([data.cos_sim[i, j] for j in range(L) if j != targets[i]], dim=0) #TODO update with S-AV integration

            positives.append(positive)
            negatives.append(negative_similarities)

        # Convert the lists to tensors
        positives = geometric_mean(torch.stack(positives), dim=1)
        negatives = torch.stack(negatives)
        data.snx_positives = positives
        data.snx_negatives = negatives
        data.snx_queries = data.q

        return data.sim_list

    def backward(self, *args, **kwargs):
        positives = self.data.snx_positives
        negatives = self.data.snx_negatives
        self.loss = self.criterion(positives, negatives)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss
