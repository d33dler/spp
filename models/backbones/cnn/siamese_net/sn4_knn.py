from dataclasses import field

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim, Tensor

from torch.nn.functional import cosine_similarity
from models.backbones.base import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.clustering import KNN_itc
from models.utilities.custom_loss import NPairMCLoss, NPairMCLossLSE
from models.utilities.utils import DataHolder, get_norm_layer, init_weights_kaiming


##############################################################################
# Class: SiameseNetworkKNN
##############################################################################

# Model: Siamese_FourLayer_64F_KNN
# Input: 3 x 84 x 84
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet, Stanford Dogs, Stanford Cars, CUB-200-2011)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class SiameseNetworkKNN(BaselineBackbone2d):
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

        # freeze batchnorm layers
        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]  # , (self.fc, [1, 4])]
        self.lr = model_cfg.LEARNING_RATE
        self.features.apply(init_weights_kaiming)
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=tuple(model_cfg.BETA_ONE))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=0.0001)
        self.criterion = NPairMCLossLSE().cuda()
        self.knn = KNN_itc(self.data.cfg.K_NEIGHBORS)

    def forward(self):
        data = self.data

        if data.is_training():
            queries = data.snx_queries
            data.snx_query_f = self.features(queries)
            data.snx_positive_f = self.features(self.data.snx_positives)
            # construct negatives out of positives for each class (N=50) so negatives = N-1
            negatives = []
            positives = data.snx_positive_f
            for i in range(len(positives)):
                negatives.append(positives[torch.arange(len(positives)) != i])
            self.data.snx_negative_f = torch.stack(negatives)
            data.sim_list = self._calc_cosine_similarities(data.snx_query_f, data.snx_positive_f, data.snx_negative_f,
                                                           data.get_true_AV(), self.knn.neighbor_k)
            return data.sim_list
        else:
            super().forward()
        return data.sim_list

    @staticmethod
    def get_topk_values(matrix: Tensor, k: int, dim: int) -> Tensor:
        return torch.topk(matrix, k, dim)[0]

    def _calc_cosine_similarities(self, queries, positives, negatives, av, neighbor_k):
        """
        Compute the cosine similarity between query and positive and negatives.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, out_channels, height, width]
        positives : torch.Tensor
            Tensor of positive embeddings of shape [batch_size // av, out_channels, height, width]
        negatives : torch.Tensor
            Tensor of negative embeddings of shape [batch_size, num_negatives, out_channels, height, width]
        av : int
            Number of augmented views.
        neighbor_k: int
            Number of nearest neighbors to consider.

        Returns
        -------
        query_pos_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and positive of shape [batch_size // av,]
        query_neg_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and negatives of shape [batch_size, num_negatives]
        """
        batch_size, out_channels, height, width = queries.size()
        unique_samples = batch_size // av
        # Initialize output tensors
        query_pos_cos_sim = torch.zeros(unique_samples).to(queries.device)
        query_neg_cos_sim = torch.zeros(unique_samples, negatives.size(1)).to(queries.device)

        # For each query in the batch
        for i in range(0, batch_size, av):
            # Compute cosine similarity with positive
            index = i // av
            positive_sam = positives[index, :, :, :].reshape(out_channels, -1)
            positive_sam_norm = torch.norm(positive_sam, dim=0, keepdim=True)
            positive_sam = positive_sam / positive_sam_norm

            for av_i in range(av):
                query_sam = queries[i + av_i, :, :, :].reshape(out_channels, -1)
                query_sam_norm = torch.norm(query_sam, dim=1, keepdim=True)
                query_sam = query_sam / query_sam_norm

                innerproduct_matrix = query_sam.t() @ positive_sam
                topk_value, _ = torch.topk(innerproduct_matrix, neighbor_k, dim=1)
                query_pos_cos_sim[index] += torch.log(torch.clamp_min(torch.sum(topk_value), min=1e-8))

                # Compute cosine similarity with negatives
                for k in range(negatives.size(1) - 1):
                    negative_sam = negatives[index, k, :, :, :].reshape(out_channels, -1)
                    negative_sam_norm = torch.norm(negative_sam, dim=0, keepdim=True)
                    negative_sam = negative_sam / negative_sam_norm

                    innerproduct_matrix = query_sam.t() @ negative_sam
                    topk_value, _ = torch.topk(innerproduct_matrix, neighbor_k, dim=1)
                    query_neg_cos_sim[index, k] += torch.log(torch.clamp_min(torch.sum(topk_value), min=1e-8))

        # Average the cosine similarities
        query_pos_cos_sim /= av
        query_neg_cos_sim /= av

        # Take the exponential of the log sum to obtain the geometric mean
        query_pos_cos_sim = torch.exp(query_pos_cos_sim)
        query_neg_cos_sim = torch.exp(query_neg_cos_sim)
        return query_pos_cos_sim, query_neg_cos_sim

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
        num_classes, num_samples_per_class, embedding_dim = support_sets.size()

        # Compute cosine similarity between each query and each sample of each support class
        class_cos_sim = cosine_similarity(queries.unsqueeze(1).unsqueeze(1), support_sets.unsqueeze(0), dim=-1)

        # Compute geometric mean for each query and the support class
        class_cos_sim = torch.prod(class_cos_sim, dim=2) ** (1.0 / num_samples_per_class)
        return class_cos_sim

    def backward(self, *args, **kwargs):
        self.loss = self.criterion(self.data.sim_list[0], self.data.sim_list[1])
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def calculate_loss(self, pred, gt):
        return torch.Tensor([0.0])
