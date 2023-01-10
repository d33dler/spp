from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys

from models.architectures.classifier import DataHolder
from models.model_utils.utils import load_config


class KNN_itc(nn.Module):
    """
    KNN-itc (Image to class) metric
    """

    def __init__(self):
        super(KNN_itc, self).__init__()
        self.neighbor_k = 0
        self.cfg = load_config(Path(__file__).parent / 'config.yaml')  # not used currently

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()
        Similarity_list = []

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.reshape((C, -1))
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j]
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

    def forward(self, data: DataHolder):
        self.neighbor_k = len(data.cfg.CLASS_NAMES)
        data.knn_list = self.cal_cosinesimilarity(data.q, data.S)

        return data
