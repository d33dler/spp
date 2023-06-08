from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim

from torch.nn.functional import cosine_similarity
from models.backbones.base import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.clustering import KNN_itc
from models.utilities.custom_loss import NPairMCLoss
from models.utilities.utils import DataHolder, get_norm_layer, init_weights_kaiming
import torch.nn.functional as F


##############################################################################
# Class: Siamese_ResNet18
##############################################################################

# Model: Siamese_FourLayer_64F
# Input: Query set, Support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class SiameseNetwork(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder, config: EasyDict = None):
        super().__init__(data, config)
        self.data = data
        model_cfg = data.cfg.BACKBONE

        self.require_grad = model_cfg.GRAD

        # norm_layer, use_bias = get_norm_layer(model_cfg.NORM)
        self.output_shape = 64
        self.fc = nn.Sequential(nn.Linear(64 * 21 * 21, 2048))
                                # nn.BatchNorm1d(4096),
                                # nn.LeakyReLU(0.2, True),
                                # nn.Dropout(p=0.3),
                                # nn.Linear(4096, 1600),
                                # nn.BatchNorm1d(1600),
                                # nn.LeakyReLU(0.2, True))

        # freeze batchnorm layers
        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]  # , (self.fc, [1, 4])]
        self.lr = model_cfg.LEARNING_RATE
        self.features.apply(init_weights_kaiming)
        self.fc.apply(init_weights_kaiming)
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0.0001)
        self.criterion = NPairMCLoss().cuda()

    def forward(self):
        data = self.data
        if data.is_training():
            queries = data.snx_queries

            data.snx_query_f = F.normalize(self.fc(self.features(queries).flatten(start_dim=1)), p=2, dim=1)
            data.snx_positive_f = F.normalize(self.fc(self.features(self.data.snx_positives).flatten(start_dim=1)), p=2,
                                              dim=1)
            # construct negatives out of positives for each class (N=50) so negatives = N-1
            negatives = []
            positives = data.snx_positive_f
            # nbatch = data.snx_queries.size(0) // self.data.cfg.EPISODE_SIZE
            # for i in range(self.data.cfg.EPISODE_SIZE):
            #     batch_slice = positives[i * nbatch: i * nbatch + nbatch]

            for j in range(0, len(positives), data.get_true_AV()):
                mask = torch.tensor([i not in range(j, j + data.get_true_AV()) for i in range(len(positives))])
                for _ in range(data.get_true_AV()):
                    negatives.append(positives[mask])
            self.data.snx_negative_f = torch.stack(negatives)

            # data.sim_list = self._calc_cosine_similarities(data.snx_query_f, data.snx_positive_f, data.snx_negative_f,
            #                                                data.get_true_AV())
            self.data.sim_list = None
            return None
        else:
            queries = data.q_in
            data.q_F = self.fc(self.features(queries).flatten(start_dim=1))
            support_sets = data.S_in
            data.S_F = torch.stack([self.fc(self.features(s_cls).flatten(start_dim=1)) for s_cls in support_sets],
                                   dim=0)
            data.sim_list = self._calc_cosine_similarities_support(data.q_F, data.S_F)
        return data.sim_list

    def _calc_cosine_similarities_support(self, queries, support_sets):
        """
        Compute the cosine similarity between each query and each sample of each support class.
        Compute geometric means for each query and the support class.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        support_sets : torch.Tensor
            Tensor of support sets of shape [num_classes, num_samples_per_class, embedding_dim]

        Returns
        -------
        class_cos_sim : torch.Tensor
            Tensor of cosine similarities between each query and each support class of shape [batch_size, num_classes]
        """
        # Compute cosine similarity between each query and each sample of each support class
        class_cos_sim = cosine_similarity(queries.unsqueeze(1).unsqueeze(1), support_sets.unsqueeze(0), dim=-1)

        # Compute arithmetic mean for each query and the support class
        class_cos_sim = torch.mean(class_cos_sim, dim=2)
        return class_cos_sim

    def backward(self, *args, **kwargs):
        data = self.data
        self.loss = self.criterion(data.snx_query_f, data.snx_positive_f, data.snx_negative_f, data.get_true_AV())
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def calculate_loss(self, pred, gt):
        return torch.Tensor([0.0])
