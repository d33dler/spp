from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim

from torch.nn.functional import cosine_similarity
from models.backbones.base import BaseBackbone2d
from models.utilities.custom_loss import NPairMCLoss
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
        self.criterion = NPairMCLoss().cuda()

    def forward(self):
        data = self.data

        if data.is_training():
            queries = data.snx_queries
            data.snx_query_f = self.fc(self.features(queries).flatten(start_dim=1))
            data.snx_positive_f = self.fc(self.features(self.data.snx_positives).flatten(start_dim=1))
            # construct negatives out of positives for each class (N=50) so negatives = N-1
            negatives = []
            positives = data.snx_positive_f
            for i in range(len(positives)):
                negatives.append(positives[torch.arange(len(positives)) != i])
            self.data.snx_negative_f = torch.stack(negatives)
            data.sim_list = self._calc_cosine_similarities(data.snx_query_f, data.snx_positive_f, data.snx_negative_f,
                                                           data.av_num)
            return None
        else:
            queries = data.q_in
            data.q_F = self.fc(self.features(queries).flatten(start_dim=1))
            support_sets = data.S_in
            data.S_F = torch.cat([self.fc(self.features(s_cls).flatten(start_dim=1)) for s_cls in support_sets], dim=0)
            data.sim_list = self._calc_cosine_similarities_support(data.q_F, data.S_F)
        return data.sim_list

    def _calc_cosine_similarities(self, queries, positives, negatives, av):
        """
        Compute the cosine similarity between query and positive and negatives.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        positives : torch.Tensor
            Tensor of positive embeddings of shape [batch_size, embedding_dim]
        negatives : torch.Tensor
            Tensor of negative embeddings of shape [batch_size, num_negatives, embedding_dim]
        av : int
            Number of augmented views. If av = 0, then just compute the distance without averaging.

        Returns
        -------
        query_pos_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and positive of shape [batch_size,]
        query_neg_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and negatives of shape [batch_size, num_negatives]
        """
        batch_size = queries.size(0)

        # Compute cosine similarity between query and positive
        query_pos_cos_sim = cosine_similarity(queries, positives)

        # Compute cosine similarity between query and negatives
        query_neg_cos_sim = cosine_similarity(queries.unsqueeze(1), negatives, dim=-1).squeeze(1)

        if av > 0:
            # If av > 0, we reshape the cosine similarities for each sample in the batch
            # Then we take the geometric mean across the augmented views
            query_pos_cos_sim = query_pos_cos_sim.view(batch_size // av, av)
            query_neg_cos_sim = query_neg_cos_sim.view(batch_size // av, av, -1)

            query_pos_cos_sim = torch.prod(query_pos_cos_sim, dim=1) ** (1.0 / av)
            query_neg_cos_sim = torch.prod(query_neg_cos_sim, dim=1) ** (1.0 / av)

        return query_pos_cos_sim, query_neg_cos_sim

    def _calc_cosine_similarities_support(self, queries, support):
        """
        Compute the cosine similarity between query and each class in the support set.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        support : torch.Tensor
            Tensor of support set embeddings of shape [num_classes, num_samples, embedding_dim]

        Returns
        -------
        cos_sim : torch.Tensor
            Tensor of cosine similarities between queries and support set of shape [batch_size, num_classes]
        """
        num_classes, num_samples, _ = support.size()

        # Compute cosine similarity between queries and support set
        cos_sim = cosine_similarity(queries.unsqueeze(1).unsqueeze(1), support, dim=-1)

        # Compute geometric mean for each class
        cos_sim = cos_sim.view(-1, num_classes, num_samples)
        cos_sim = torch.prod(cos_sim, dim=2) ** (1.0 / num_samples)

        return cos_sim

    def backward(self, *args, **kwargs):
        queries = self.data.snx_query_f
        positives = self.data.snx_positive_f
        negatives = self.data.snx_negative_f
        self.loss = self.criterion(positives, negatives, torch.cat((queries, positives), 0))
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def adjust_learning_rate(self, epoch):
        self.scheduler.step()
