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

    def cosine_similarity(self, q: Tensor, S: Tensor, qAV_num: int = 1, SAV_num: int = 1, strategy: str = 'N:1',
                          shot_num=1):
        """
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class
        """

        # input1---query images
        input1 = q.contiguous().view(q.size(0), q.size(1), -1)  # (batchsize, 64, 441)
        input1 = input1.permute(0, 2, 1)  # (batchsize, 441, 64)
        # input2--support set
        input2 = S.contiguous().view(S.size(0), S.size(1), -1)  # 25 * 64 * 441
        input2 = input2.permute(0, 2, 1)  # 25 * 441 * 64

        # L2 Normalization
        input1_norm = torch.norm(input1, 2, 2, True)  # (batchsize, AV_count, 441, 1)
        query = input1 / input1_norm  # (batchsize, AV_count, 441, 64)

        input2_norm = torch.norm(input2, 2, 2, True)  # 25 * 441 * 1
        support_set = input2 / input2_norm  # 25 * 441 * 64
        support_set = support_set.contiguous().view(-1, SAV_num * shot_num * support_set.size(1),
                                                    support_set.size(2))  # 5 * x * 64
        support_set = support_set.permute(0, 2, 1)  # 5 * 64 * x

        # cosine similarity between a query set and a support set
        innerproduct_matrix = torch.matmul(query.unsqueeze(1), support_set)  # (batchsize, AV_count, 5, 441, 2205)
        # reshape innerproduct into augmented views sets of each query
        B, L, *_ = innerproduct_matrix.size()
        innerproduct_matrix = innerproduct_matrix.contiguous() \
            .view(B // qAV_num, qAV_num, innerproduct_matrix.size(1),
                  innerproduct_matrix.size(2),
                  innerproduct_matrix.size(
                      3))  # (batchsize, AV_count, 5, 441, 2205)
        # choose the top-k nearest neighbors
        topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k,
                                            -2)  # (batchsize, AV_count, 5, 441, 3)

        img2class_sim = torch.sum(torch.sum(topk_value, -1), -1)  # (batchsize, AV_count, 5)
        # geometric mean
        self.similarity_ls = self._geometric_mean(img2class_sim, dim=1)  # (batchsize, 5)
        self.topk_cosine_sums = None
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

    def forward(self, q, S, qAV_num=1, SAV_num=1, strategy='N:1', shot_num=1):
        return self.cosine_similarity(q, S, qAV_num, SAV_num, 'N:1' if strategy is None else strategy, shot_num)
