import functools
import os
from enum import Enum, EnumMeta
from typing import Dict, List

import torch
from easydict import EasyDict
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from data_loader.data_load import Parameters, DatasetLoader
from models.utilities.utils import save_checkpoint, load_config, accuracy


class ArchM(nn.Module):
    """
    Architecture Module - wraps around nn.Module and organizes necessary functions/values
    """

    class Child(nn.Module):
        lr: float | List[float]
        optimizer: Optimizer | List[Optimizer]
        criterion: _Loss | List[_Loss]
        loss: Tensor
        optimize = True

        def calculate_loss(self, pred, gt):
            if isinstance(self.criterion, list):
                loss_ls = []
                for prediction, criterion, optimizer in zip(pred, self.criterion, self.optimizer):
                    loss = criterion(prediction, gt)
                    loss_ls.append(loss)
                return loss_ls
            return self.criterion(pred, gt)

        def backward(self, pred, gt):
            if not self.training:
                return
            if isinstance(self.criterion, list):
                loss_ls = []
                for prediction, criterion, optimizer in zip(pred, self.criterion, self.optimizer):
                    loss = criterion(prediction, gt)
                    loss_ls.append(loss)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                torch.cuda.empty_cache()
                self.loss = loss_ls
                return

            self.loss = self.criterion(pred, gt)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        def adjust_learning_rate(self, epoch):

            if isinstance(self.optimizer, list):
                for lr, optimizer in zip(self.lr, self.optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (0.05 ** (epoch // 10))
            else:
                lr = self.lr * (0.05 ** (epoch // 10))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        def set_optimize(self, optimize=True):
            self.optimize = optimize

        def load_optimizer_state_dict(self, optim_state_dict):
            if isinstance(self.optimizer, list):
                for optimizer, state in zip(self.optimizer, optim_state_dict):
                    optimizer.load_state_dict(state)
            else:
                self.optimizer.load_state_dict(optim_state_dict)

    class BaseConfig:
        __doc__ = "Base config class for yaml files. Override __doc__ for implementations."

    module_topology: Dict[str, Child]
    model_cfg: EasyDict
    best_prec1 = 0
    epochix = 0
    optimizers: dict

    def __init__(self, cfg_path) -> None:
        super().__init__()
        self.model_cfg = self.load_config(cfg_path)
        self.module_topology: Dict[str, ArchM.Child] = {_: None for _ in self.model_cfg.TOPOLOGY}
        c = self.model_cfg
        p = Parameters(c.IMAGE_SIZE, c.DATASET_DIR, c.SHOT_NUM, c.WAY_NUM, c.QUERY_NUM, c.EPISODE_TRAIN_NUM,
                       c.EPISODE_TEST_NUM, c.EPISODE_VAL_NUM, c.OUTF, c.WORKERS, c.EPISODE_SIZE,
                       c.TEST_EPISODE_SIZE, c.BATCH_SIZE)
        self.data_loader = DatasetLoader(c.AUGMENTOR, p)

    class ActivationFuncs(Enum):
        Relu = nn.ReLU
        Lrelu = nn.LeakyReLU
        Sigmoid = nn.Sigmoid

    class NormalizationFuncs(Enum):
        BatchNorm2d = functools.partial(nn.BatchNorm2d, affine=True)
        InstanceNorm2d = functools.partial(nn.InstanceNorm2d, affine=False)
        none = None

    class PoolingFuncs(Enum):
        MaxPool2d = nn.MaxPool2d
        AveragePool2d = nn.AvgPool2d
        LPPool2d = nn.LPPool2d

    @staticmethod
    def get_func(fset: EnumMeta, name: str):
        if name not in fset.__members__.keys():
            raise NotImplementedError('Function [%s] not found' % str)
        return fset[name].value

    def _set_modules_mode(self):
        for k, m in self.module_topology.items():
            v = self.model_cfg[k].MODE == 'TRAIN'
            print("Setting module: ", k, " TRAIN" if v else " TEST", " mode.")
            m.train(v)

    def save_model(self, filename):
        state = {
            'epoch_index': self.epochix,
            'arch': self.arch,
            'best_prec1': self.best_prec1
        }
        optimizers = {
            f"{k}_optim": v.optimizer.state_dict() if not isinstance(v.optimizer, list) else
            [o.state_dict() for o in v.optimizer] for k, v in self.module_topology.items() if v.optimizer is not None}
        state_dicts = {f"{k}_state_dict": v.state_dict() for k, v in self.module_topology.items()}
        state.update(optimizers)
        state.update(state_dicts)
        save_checkpoint(state, filename)

    def load_model(self, path, txt_file=None):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self.epochix = checkpoint['epoch_index']
            self.best_prec1 = checkpoint['best_prec1']
            [v.load_state_dict(checkpoint[f"{k}_state_dict"]) for k, v in self.module_topology.items()
             if f"{k}_state_dict" in checkpoint]
            [v.load_optimizer_state_dict(checkpoint[f"{k}_optim"]) for k, v in self.module_topology.items()
             if f"{k}_optim" in checkpoint]
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']))
            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']), file=txt_file)
        else:
            print("=> no checkpoint found at '{}'".format(path))
            print("=> no checkpoint found at '{}'".format(path), file=self.p)

    def load_config(self, path):
        self.model_cfg = load_config(path)
        return self.model_cfg

    def get_loss(self, module_id: str):
        if module_id not in self.module_topology.keys():
            raise KeyError(f"Module {module_id} not found in architecture topology")
        return self.module_topology[module_id].loss

    def calculate_accuracy(self, output, target, topk=(1, 3)):
        prec1, _ = accuracy(output, target, topk=topk)
        self.best_prec1 = prec1 if prec1 > self.best_prec1 else self.best_prec1
        return prec1

    def adjust_learning_rate(self, epoch_num):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        for k, mod in self.module_topology.items():
            if mod.training and mod.optimize:
                mod.adjust_learning_rate(epoch=epoch_num)
