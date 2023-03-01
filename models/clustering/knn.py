from typing import List

import torch
from torch import Tensor


class KNN_itc:
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int):
        super(KNN_itc, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.neighbor_k = k_neighbors

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, q: Tensor, S: List[Tensor]):
        B, C, h, w = q.size()
        similarity_ls = []
        topk_ls = torch.zeros(B, self.neighbor_k * len(S))
        nk = self.neighbor_k
        for i in range(B):
            topk_q = torch.zeros(self.neighbor_k * len(S))
            query_sam = q[i]
            query_sam = query_sam.reshape((C, -1))
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            inner_sim = torch.zeros(1, len(S)).cuda()

            for j in range(len(S)):
                support_set_sam = S[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam @ support_set_sam
                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)
                topk_ls[i, j * nk:j * nk + nk] = torch.sum(topk_value, dim=0)

            similarity_ls.append(inner_sim)

        similarity_ls = torch.cat(similarity_ls, 0)
        return similarity_ls, topk_ls

    def forward(self, q, S):
        return self.cal_cosinesimilarity(q, S)
