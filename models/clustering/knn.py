from typing import List

import torch
from torch import Tensor, nn


class KNN_itc(nn.Module):
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int):
        super(KNN_itc, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.neighbor_k = k_neighbors

    def _geometric_mean(self, t: Tensor) -> Tensor:
        log_tensor = torch.log(t)
        mean = torch.mean(log_tensor, dim=0, keepdim=True)
        geom_mean = torch.exp(mean)
        return geom_mean

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cosine_similarity(self, q: Tensor, S: List[Tensor], av_num: int = 1):
        B, C, h, w = q.size()
        similarity_ls = []
        print("FEATURES")
        print(q.size())
        print(S[0].size())
        print("----END FEATURES----")
        for i in range(0, B, av_num):
            inner_sim = torch.zeros(av_num, len(S)).cuda()
            for j in range(len(S)):
                support_set_sam = S[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm
                for av in range(av_num):
                    query_sam = q[i + av]
                    query_sam = query_sam.reshape((C, -1))
                    query_sam = torch.transpose(query_sam, 0, 1)
                    query_sam_norm = torch.norm(query_sam, 2, 1, True)
                    query_sam = query_sam / query_sam_norm

                    # cosine similarity between a query sample and a support category
                    innerproduct_matrix = query_sam @ support_set_sam
                    # choose the top-k nearest neighbors
                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                    inner_sim[av, j] = torch.sum(topk_value)

            similarity_ls.append(self._geometric_mean(inner_sim) if av_num > 1 else inner_sim)

        similarity_ls = torch.cat(similarity_ls, 0)
        return similarity_ls

    def forward(self, q, S, av_num):
        return self.cosine_similarity(q, S, av_num)
