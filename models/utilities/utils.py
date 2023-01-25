"""
General utilities
"""
from __future__ import print_function

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import torch
import yaml
from easydict import EasyDict
from pandas import DataFrame
from torch import Tensor


def load_config(config: Path):
    with open(config, mode="r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return EasyDict(cfg)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


@dataclass
class DataHolder:
    """
    Module IO object
    """

    # Input
    q_in: Tensor
    S_in: list
    # Backbone2d-out
    q: Tensor  # TODO identify types
    S: list
    # KNN-out
    knn_list: List[List]
    knn_raw: List[List]
    q_N: Tensor
    # Autoencoder
    q_enc: Tensor
    S_enc: List[Tensor]
    # Tree fit input
    X: DataFrame
    y: DataFrame
    eval_set: Tuple[Any, Any]

    # Tree-out
    tree_pred: np.ndarray

    def __init__(self, cfg):
        self.eval_set = None
        self.module_list: List = []
        self.cfg = cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape((1, -1)).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
