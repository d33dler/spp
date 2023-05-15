from typing import List

import torch
from torch import Tensor, nn


class KNN_itc(nn.Module):
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int):
        super(KNN_itc, self).__init__()
        self.neighbor_k = k_neighbors

    def _geometric_mean(self, t: Tensor) -> Tensor:
        log_tensor = torch.log(t)

        if torch.isnan(log_tensor).any():
            print(t)
            print(">TNSR ------ LOG ISNA: ", torch.isnan(log_tensor).any())
            print("> HAS0: ", (t == 0).any())
            print("> HASNAN: ", torch.isnan(t).any())
            print("> HASINF: ", torch.isinf(t).any())
            print("> HASNEG: ", (t < 0).any())
            exit(0)
        mean = torch.mean(log_tensor, dim=0, keepdim=True)
        geom_mean = torch.exp(mean)
        return geom_mean

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cosine_similarity(self, q: Tensor, S: List[Tensor], av_num: int = 1):
        B, C, h, w = q.size()
        similarity_ls = []
        topk_cosine_sums = []

        for i in range(0, B, av_num):
            inner_sim = torch.zeros(av_num, len(S)).cuda()
            topk_cosine_sum = torch.zeros(av_num, len(S), S[0].size(1) // 441).cuda()

            for j in range(0, len(S)):
                support_set_sam = S[j]  # support set AV
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / (support_set_sam_norm + 1e-8)

                for av in range(av_num):
                    query_sam = q[i + av]  # query sample AV
                    query_sam = query_sam.reshape((C, -1))
                    query_sam = torch.transpose(query_sam, 0, 1)
                    query_sam_norm = torch.norm(query_sam, 2, 1, True)
                    query_sam: Tensor = query_sam / (query_sam_norm + 1e-8)

                    # cosine similarity between a query sample and a support category
                    innerproduct_matrix = query_sam @ support_set_sam

                    # split the tensor and calculate top-k sums for each shot separately
                    innerproduct_matrix_split = torch.split(innerproduct_matrix, 441, dim=1)
                    topk_cosine_sum[av, j] = torch.stack(
                        [torch.sum(torch.topk(v, self.neighbor_k, dim=1)[0]) for v in innerproduct_matrix_split])

                    # choose the top-k nearest neighbors
                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)

                    # sum up the cosine similarity of the top-k nearest neighbors
                    inner_sim[av, j] = torch.sum(topk_value)
            similarity_ls.append(self._geometric_mean(inner_sim) if av_num > 1 else inner_sim)
            # print(">TOPK COSINE SUM HAS0: ", (topk_cosine_sum == 0).any())
            topk_cosine_sum = self._geometric_mean(torch.clamp_min(topk_cosine_sum, 1e-8)) if av_num > 1 else inner_sim
            topk_cosine_sums.append(topk_cosine_sum)
            # print(">TOPK COSINE SUM ISNA: ", torch.isnan(topk_cosine_sum).any())

        similarity_ls = torch.cat(similarity_ls, 0)
        topk_cosine_sums = torch.cat(topk_cosine_sums, 0)
        return similarity_ls, topk_cosine_sums

    def forward(self, q, S, av_num):
        return self.cosine_similarity(q, S, av_num)
