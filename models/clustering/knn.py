from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
import functools
import pdb
import math
import sys

from models.utilities.utils import load_config, DataHolder


class KNN_itc:
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self, k_neighbors: int):
        super(KNN_itc, self).__init__()

        self.neighbor_k = k_neighbors

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, q: Tensor, S: List[Tensor]):
        B, C, h, w = q.size()
        Similarity_list = []
        for i in range(B):
            query_sam = q[i]
            query_sam = query_sam.reshape((C, -1))
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            if torch.cuda.is_available():
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

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list

    def forward(self, q, S):
        sim_list = self.cal_cosinesimilarity(q, S)
        return sim_list
