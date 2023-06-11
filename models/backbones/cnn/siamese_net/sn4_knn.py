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
from torch.nn.functional import softmax


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
        self.criterion = NPairMCLoss().cuda()
        self.knn = KNN_itc(self.data.cfg.K_NEIGHBORS)

    def forward(self):
        data = self.data

        if data.is_training():
            queries = data.snx_queries
            data.snx_query_f = self.features(queries)
            data.snx_positive_f = self.features(self.data.snx_positives)
            # construct negatives out of positives for each class (N=classes * av) therefore, negatives = (N-1) * av
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

        # For each unique sample in the batch
        for i in range(0, batch_size, av):
            # Compute all-to-all cosine similarity with positive
            index = i // av
            positive_sam_all = positives[i: i + av, :, :, :].reshape(av, out_channels, -1)
            positive_sam_all_norm = torch.norm(positive_sam_all, dim=1, keepdim=True)
            positive_sam_all = positive_sam_all / positive_sam_all_norm

            for av_i in range(av):
                query_sam = queries[i + av_i, :, :, :].reshape(out_channels, -1).t()
                query_sam_norm = torch.norm(query_sam, dim=1, keepdim=True)
                query_sam = query_sam / query_sam_norm

                for av_j in range(av):
                    positive_sam = positive_sam_all[av_j, :, :].reshape(out_channels, -1)
                    innerproduct_matrix = query_sam @ positive_sam
                    topk_value, _ = torch.topk(innerproduct_matrix, neighbor_k, dim=1)
                    query_pos_cos_sim[index] += torch.sum(topk_value)

                # Compute cosine similarity with negatives (considering only the current view of anchor)
                for k in range(negatives.size(1)):
                    negative_sam = negatives[index, k, :, :, :].reshape(out_channels, -1)
                    negative_sam_norm = torch.norm(negative_sam, dim=0, keepdim=True)
                    negative_sam = negative_sam / negative_sam_norm

                    innerproduct_matrix = query_sam @ negative_sam
                    topk_value, _ = torch.topk(innerproduct_matrix, neighbor_k)
                    query_neg_cos_sim[index, k] += torch.sum(topk_value)

        # Average the cosine similarities
        query_pos_cos_sim /= (av ** 2)
        query_neg_cos_sim /= av

        # Scaling the cosine similarities [-1 ... 1]
        similarities = torch.cat([query_pos_cos_sim.unsqueeze(1), query_neg_cos_sim], dim=1)
        max_val_per_tuple = torch.max(similarities, dim=1, keepdim=True)[0]
        normalized_similarities = similarities / max_val_per_tuple
        query_pos_cos_sim = normalized_similarities[:, 0]
        query_neg_cos_sim = normalized_similarities[:, 1:]
        return query_pos_cos_sim, query_neg_cos_sim

    def backward(self, *args, **kwargs):
        data = self.data
        self.loss = self.criterion(data.snx_query_f, data.snx_positive_f, data.snx_negative_f,
                                   data.get_true_AV(), data.sim_list[0], data.sim_list[1])
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def calculate_loss(self, pred, gt):
        return torch.Tensor([0.0])
