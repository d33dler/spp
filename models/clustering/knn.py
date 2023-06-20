from typing import List

import torch
from torch import Tensor, nn


class KNN_itc(nn.Module):
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int, compute_cos_N=False):
        super(KNN_itc, self).__init__()
        self.similarity_ls = []
        self.topk_cosine_sums = []
        self.neighbor_k = k_neighbors
        self.compute_cos_N = compute_cos_N
        self.strategy_map = {
            "1:1": self._strategy_ovo,
            "N:1": self._strategy_rvo,
            "N:N": self._strategy_rvr,
        }

    @staticmethod
    def _geometric_mean(t: Tensor) -> Tensor:
        log_tensor = torch.log(t)
        mean = torch.mean(log_tensor, dim=0, keepdim=True)
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

    def cosine_similarity(self, q: Tensor, S: List[Tensor], qAV_num: int = 1, SAV_num: int = 1, strategy: str = 'N:1'):
        """
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class
        """

        B, _, _ = q.size()

        self.similarity_ls = []
        self.topk_cosine_sums = []

        for i in range(0, B, qAV_num):
            inner_sim, topk_cosine_sum = self.strategy_map[strategy](q, S, qAV_num, SAV_num, i)

            self.similarity_ls.append(self.apply_geometric_mean(inner_sim, qAV_num))
            if self.topk_cosine_sums:
                topk_cosine_sum = self.apply_geometric_mean(topk_cosine_sum, qAV_num)
                self.topk_cosine_sums.append(topk_cosine_sum)

        self.similarity_ls = torch.cat(self.similarity_ls, 0)

        self.topk_cosine_sums = torch.cat(self.topk_cosine_sums, 0) if self.compute_cos_N else None
        return self.similarity_ls, self.topk_cosine_sums

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
                if self.compute_cos_N:
                    topk_cosine_sum[av, cls_ix] = self.get_topk_cosine_sum(innerproduct_matrix)
                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)

                inner_sim[av, cls_ix] = torch.sum(topk)
        return inner_sim, topk_cosine_sum

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
        topk_cosine_sum = None
        # if self.compute_cos_N:
        #     topk_cosine_sum = torch.zeros(qAV_num, len(S), S[0].size(1) // 441).cuda()
        for j in range(len(S)):
            support_set_sam = self.normalize_tensor(S[j], 0)  # support set AV

            for av in range(qAV_num):
                query_sam = self.normalize_tensor(q[i + av], 1)  # query sample AV
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                # if self.compute_cos_N:
                #     topk_cosine_sum[av, j] = self.get_topk_cosine_sum(innerproduct_matrix)
                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[av, j] = torch.sum(topk)
        return inner_sim, topk_cosine_sum

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
        topk_cosine_sum = None
        if self.compute_cos_N:
            topk_cosine_sum = torch.zeros(qAV_num * SAV_num, cls_size, S[0].size(1) // 441).cuda()
        index = [0 for _ in range(cls_size)]
        for k in range(qAV_num):

            query_sam = self.normalize_tensor(q[i + k], 1)  # query sample AV
            for j in range(0, len(S), SAV_num):
                cls_ix = j // SAV_num
                for av in range(SAV_num):
                    support_set_sam = self.normalize_tensor(S[j + av], 0)
                    # cosine similarity between a query sample and a support category

                    innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                    if self.compute_cos_N:
                        topk_cosine_sum[index[cls_ix], cls_ix] = self.get_topk_cosine_sum(innerproduct_matrix)
                    inner_sim[index[cls_ix], cls_ix] = torch.sum(
                        self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1))
                    index[cls_ix] += 1
        return inner_sim, topk_cosine_sum

    def forward(self, q, S, qAV_num=1, SAV_num=1, strategy='N:1'):
        return self.cosine_similarity(q, S, qAV_num, SAV_num, 'N:1' if strategy is None else strategy)
