from typing import List

import torch
from torch import Tensor, nn


class I2C_KNN(nn.Module):
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int, classes=5, ):
        super(I2C_KNN, self).__init__()
        self.similarity_ls = []
        self.topk_cosine_sums = []
        self.neighbor_k = k_neighbors
        self.strategy_map = {
            "1:1": self._strategy_ovo,
            "N:1": self._strategy_rvo,
            "N:N": self._strategy_rvr,
        }
        self.classes = classes

    @staticmethod
    def _geometric_mean(t: Tensor, dim=0) -> Tensor:
        log_tensor = torch.log(t)
        mean = torch.mean(log_tensor, dim=dim, keepdim=False)
        geom_mean = torch.exp(mean)
        return geom_mean

    @staticmethod
    def normalize_tensor(tensor: Tensor, dim: int) -> Tensor:
        tensor_norm = torch.norm(tensor, 2, dim, True)
        return tensor / tensor_norm

    @staticmethod
    def get_cosine_similarity(query: Tensor, support: Tensor) -> Tensor:
        return query @ support

    def get_topk_cosine_sum(self, matrix: Tensor, split_dim=1, topk_dim: int = 1) -> Tensor:
        matrix_split = torch.split(matrix, 441, dim=split_dim)
        return torch.stack([torch.sum(torch.topk(v, self.neighbor_k, dim=topk_dim)[0]) for v in matrix_split])

    @staticmethod
    def get_topk_values(matrix: Tensor, k: int, dim: int) -> Tensor:
        return torch.topk(matrix, k, dim)[0]

    def apply_geometric_mean(self, tensor: Tensor, av_num: int) -> Tensor:
        return self._geometric_mean(torch.clamp_min(tensor, 1e-8)) if av_num > 1 else tensor

    def cosine_similarity(self, q: Tensor, S: Tensor, av_num: int = 1, **kwargs) -> Tensor:
        """
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class
        """

        # Reshape and permute query and support set tensors
        query = self._reshape_and_normalize(q)
        support_set = self._reshape_and_normalize(S)

        support_set = support_set.contiguous().view(1, -1, support_set.size(2))  # 1 * x * 64
        support_set = support_set.permute(0, 2, 1)  # 1 * 64 * x
        # Compute cosine similarity between query and support set
        innerprod_mx = torch.matmul(query.unsqueeze(1), support_set)

        # Reshape innerproduct into augmented views sets of each query
        B, *_ = innerprod_mx.size()
        innerprod_mx = innerprod_mx.squeeze()
        innerprod_mx = innerprod_mx.view(B // av_num, av_num, innerprod_mx.size(1),
                                         self.classes, innerprod_mx.size(2) // self.classes)

        # Choose the top-k nearest neighbors
        topk_value, topk_index = torch.topk(innerprod_mx, self.neighbor_k, -1)

        # Compute image-to-class similarity
        img2class_sim = self._compute_img2class_sim(topk_value, **kwargs)

        # Aggregate the similarity values of all augmented views of each query
        self.similarity_ls = self._geometric_mean(img2class_sim, dim=1)

        return self.similarity_ls

    def _reshape_and_normalize(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x_norm = torch.norm(x, 2, 2, True)
        return x / x_norm

    def _compute_img2class_sim(self, topk_value: Tensor, **kwargs) -> Tensor:
        if self.neighbor_k > 1:
            img2class_sim = torch.sum(torch.sum(topk_value, -1), -2)
        else:
            img2class_sim = torch.sum(topk_value.squeeze(-1), -2)
        return torch.clamp_min(img2class_sim, 1e-8)

    def _strategy_ovo(self, q, S, qAV_num=1, SAV_num=1, i=0):
        """
        1:1 strategy : 1 query AV, 1 support class AV-subset
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class

        """
        inner_sim = torch.zeros(qAV_num, len(S) // SAV_num).cuda()
        topk_cosine_sum = None
        if self.compute_cos_N:
            topk_cosine_sum = torch.zeros(qAV_num, len(S) // SAV_num, S[0].size(1) // S[0]).cuda()
        for j in range(0, len(S), qAV_num):
            cls_ix = j // SAV_num
            for av in range(qAV_num):
                support_set_sam = self.normalize_tensor(S[j + av], 0)  # support set AV

                query_sam = self.normalize_tensor(q[i + av], 1)  # query sample AV
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)

                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[av, cls_ix] = torch.sum(topk)
        return inner_sim

    def _strategy_rvo(self, q, S, qAV_num=1, SAV_num=None, i=0):
        """
        N:1 strategy : N query AVs, 1 support clas
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: ignored, used for consistency
        param i: index of query sample
        """
        inner_sim = torch.zeros(qAV_num, len(S)).cuda()
        for j in range(len(S)):
            support_set_sam = self.normalize_tensor(S[j], 0)  # support set AV

            for av in range(qAV_num):
                query_sam = self.normalize_tensor(q[i + av], 1)  # query sample AV
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[av, j] = torch.sum(topk)
        return inner_sim

    def _strategy_rvr(self, q, S, qAV_num=1, SAV_num=1, i=0):
        """
        N:N strategy : N query AVs vs N support AVs
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class
        param i: index of query sample
        """
        cls_size = len(S) // SAV_num
        inner_sim = torch.zeros(qAV_num * SAV_num, cls_size).cuda()
        index = [0 for _ in range(cls_size)]
        for k in range(qAV_num):

            query_sam = self.normalize_tensor(q[i + k], 1)  # query sample AV
            for j in range(0, len(S), SAV_num):
                cls_ix = j // SAV_num
                for av in range(SAV_num):
                    support_set_sam = self.normalize_tensor(S[j + av], 0)
                    # cosine similarity between a query sample and a support category

                    innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                    inner_sim[index[cls_ix], cls_ix] = torch.sum(
                        self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1))
                    index[cls_ix] += 1
        return inner_sim

    def forward(self, q, S, av_num=1, SAV_num=1, strategy='N:1', shot_num=1):
        strategy = strategy if strategy is not None else 'N:1'
        self.classes = len(S) // SAV_num
        return self.cosine_similarity(q, S, av_num)


class I2C_KNN_AM(I2C_KNN):

    def __init__(self, neighbor_k, classes=5, _attention_func=None):
        super().__init__(neighbor_k, classes)
        self._attention_func = _attention_func

    def _compute_img2class_sim(self, topk_value, **kwargs):
        """
        Compute image to class similarity using M(x) Class-related Attention Mechanism
        :param topk_value: topk values
        :return: image to class similarity
        """
        x = kwargs['x']
        av = kwargs['av']
        att_x = self._attention_func(x).flatten(start_dim=2)  # (B * AV, 1, H, W)
        if self.neighbor_k == 1:
            img2class_sim = topk_value
        else:
            img2class_sim = torch.sum(topk_value, -1)  # (B, AV, HW, L)

        img2class_sim = img2class_sim.reshape(-1, img2class_sim.size(2), img2class_sim.size(3))  # (B * AV, HW, L)
        img2class_sim = torch.bmm(att_x, img2class_sim) # (B * AV, 1, L)

        img2class_sim = img2class_sim.reshape(-1, av, img2class_sim.size(2))
        return img2class_sim

    def forward(self, q, S, av_num=1, SAV_num=1, strategy='N:1', shot_num=1):
        return self.cosine_similarity(q, S, av_num, x=q, av=av_num)
