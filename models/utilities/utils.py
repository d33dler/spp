from __future__ import print_function
import functools
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
from torch import Tensor, nn as nn
from torch.nn import BatchNorm2d, init
import seaborn as sns

"""
General utilities
"""


def create_confusion_matrix(true: np.ndarray, pred: np.ndarray):
    cm = confusion_matrix(true, pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    ax.grid(False)
    plt.show()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_config(config: Path, root_cfg=None, ret_dict=False):
    with open(config, mode="r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if root_cfg is not None:
        config_exchange(cfg, root_cfg)
    if ret_dict:
        return EasyDict(cfg), cfg
    return EasyDict(cfg)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


@dataclass
class DataHolderBase:
    """
     Module IO object
     """

    cfg: Any
    module_list: List[nn.Module]
    k_neighbors: int  # 1 - num_classes
    use_bias: bool
    norm_layer: Any  # torch


def config_exchange(dest: dict, src: dict):
    for k, v in src.items():
        if k in dest:
            if v is None:
                src[k] = dest[k]
            else:
                dest[k].update(v)
    for k, v in dest.items():
        if k not in src:
            src[k] = v
    print(src)


@dataclass
class DataHolder(DataHolderBase):
    """
    Module IO specification object
    Models can choose to employ it for sharing information explicitly among child modules.
    """
    # Classification
    num_classes: int
    # Input
    q_in_CPU: Tensor
    q_in: Tensor
    S_in: List[Tensor]
    targets: Tensor
    av_num: int
    cos_sim: Tensor
    # Backbone2d OUTPUT
    q: Tensor
    DLD_topk: Tensor
    S: List[Tensor]  # CUDA
    sim_list: Tensor  # CUDA

    # Tree fit input
    X: DataFrame
    y: DataFrame
    eval_set: Tuple[Any, Any]

    # SNX
    snx_queries: Tensor
    snx_positives: Tensor

    snx_query_embeddings: Tensor
    snx_positive_embeddings: Tensor
    snx_negative_embeddings: Tensor
    # Tree-out
    tree_pred: np.ndarray

    output: Any

    def __init__(self, cfg):
        self.training = True
        self.eval_set = None
        self.module_list: List = []
        self.cfg = cfg
        self.k_neighbors = cfg.K_NEIGHBORS
        self.use_bias = cfg.USE_BIAS
        self.norm_layer = BatchNorm2d
        self.num_classes = cfg.WAY_NUM
        self.av_num = cfg.AUGMENTOR.AV_NUM or 0

    def empty_cache(self):
        del self.q_in
        del self.S_in
        del self.q
        del self.S
        del self.output
        del self.cos_sim
        gc.collect()
        torch.cuda.empty_cache()

    def get_true_AV(self):
        return self.av_num + 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.loss_list = []
        self.loss_history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.loss_list = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.loss_list.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_loss_history(self):
        self.loss_history.append(np.mean(self.loss_list))
        self.loss_list = []
        return self.loss_history


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


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    use_bias = norm_layer.func == nn.InstanceNorm2d

    return norm_layer, use_bias


def geometric_mean(t: Tensor, dim=0, keepdim=False) -> Tensor:
    return torch.exp(torch.mean(torch.log(t), dim=dim, keepdim=keepdim))


def identity(x):
    return x


def deep_convert_easydict(layer):
    to_ret = layer
    if isinstance(layer, EasyDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_easydict(value)
    except AttributeError:
        pass

    return to_ret
