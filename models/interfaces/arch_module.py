import functools
import os
from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Dict, List, Union, Any

import torch
from easydict import EasyDict
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from data_loader.data_load import Parameters, DatasetLoader
from models.utilities.utils import save_checkpoint, load_config, accuracy, init_weights, config_exchange
import torch.nn.functional as F


class ARCH(nn.Module):
    """
    Module architecture class - wraps around nn.Module and organizes necessary functions/variables and provides necessary
    model functionalities. This class also hosts an inner class ARCH.Child to be implemented by any module used in
    a model - ARCH stores the module instances and offers module management functionalities.
    Functionalities:
    Save & Load model
    Override child module configuration (root config child-module sub-configs fields matching child module configuration fields)
    Return calculated loss
    Calculate accuracy
    LR tweaking
    Any model in this framework must subclass Class.ARCH !
    """
    root_cfg: EasyDict | dict
    optimizers: Dict
    best_prec1 = 0
    _epochix = 0
    _loss_val: Tensor

    class Child(nn.Module, ABC):
        """
        ARCH.Child class - abstract class wrapping around nn.Module and providing typical
        module training functionalities. Any model sub-classing ARCH should employ modules
        sub-classing ARCH.Child.
        Functionalities:
        Loss calculation
        Backward call
        LR adjustment
        """
        config: Dict | EasyDict
        lr: float | List[float]
        optimizer: Optimizer | List[Optimizer]
        criterion: _Loss | List[_Loss]
        loss: Tensor
        fine_tuning = True
        require_grad = False

        def __init__(self, config: EasyDict | dict) -> None:
            super().__init__()
            self.config = config


        def calculate_loss(self, gt, pred):
            """
            Calculates without calculating the gradient
            :param gt: ground truth
            :type gt: Sequence
            :param pred: predictions
            :type pred: Sequence
            :return: loss
            :rtype: Any
            """
            if isinstance(self.criterion, list):
                loss_ls = []
                for prediction, criterion, optimizer in zip(pred, self.criterion, self.optimizer):
                    loss = criterion(prediction, gt)
                    loss_ls.append(loss)
                return loss_ls
            return self.criterion(pred, gt)

        def backward(self, pred, gt):
            if not (self.training and self.require_grad):
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
                return loss_ls
            self.loss = self.criterion(pred, gt)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            return self.loss

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
            self.fine_tuning = optimize

        def load_optimizer_state_dict(self, optim_state_dict):
            if isinstance(self.optimizer, list):
                for optimizer, state in zip(self.optimizer, optim_state_dict):
                    optimizer.load_state_dict(state)
            else:
                self.optimizer.load_state_dict(optim_state_dict)

        @staticmethod
        @abstractmethod
        def get_config():
            return None

        def load(self, model: Any) -> None:
            pass

        def dump(self) -> Any:
            return None

    class BaseConfig:
        __doc__ = "Base config class for yaml files. Override __doc__ for implementations."

    module_topology: Dict[str, Child]

    def __init__(self, cfg_path) -> None:
        super().__init__()
        self._store_path = None
        self.state = dict()
        self.root_cfg = self.load_config(cfg_path)
        self.module_topology: Dict[str, ARCH.Child] = self.root_cfg.TOPOLOGY
        self._mod_topo_private = self.module_topology.copy()
        c = self.root_cfg
        p = Parameters(c.IMAGE_SIZE, c.SHOT_NUM, c.WAY_NUM, c.QUERY_NUM, c.EPISODE_TRAIN_NUM,
                       c.EPISODE_TEST_NUM, c.EPISODE_VAL_NUM, c.OUTF, c.WORKERS, c.EPISODE_SIZE,
                       c.TEST_EPISODE_SIZE, c.QUERY_NUM * c.WAY_NUM)
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

    def backward(self, pred, gt):
        for m in self.module_topology.values():
            if m.training:
                m.backward(pred, gt)

    @staticmethod
    def get_func(fset: EnumMeta, name: str):
        if name not in fset.__members__.keys():
            raise NotImplementedError('Function [%s] not found' % str)
        return fset[name].value

    def _set_modules_mode(self):
        for k, m in self.module_topology.items():
            v = self.root_cfg[k].MODE == 'TRAIN'
            print("Setting module: ", k, " TRAIN" if v else " TEST", " mode.")
            m.train(v)

    def save_model(self, filename=None):
        """
        Saves model to filename or back to same checkpoint file loaded into the model.
        Subclasses can store in the ARCH.state field additional components  and call this function
        to save everything.
        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        if filename is None:
            filename = self._store_path
            if self._store_path is None:
                raise ValueError("Missing model save path!")

        priv = self._mod_topo_private

        state = {
            'epoch_index': self._epochix,
            'arch': self.arch,
            'best_prec1': self.best_prec1,

        }
        optimizers = {
            f"{priv[k]}_optim": v.optimizer.state_dict() if not isinstance(v.optimizer, list) else
            [o.state_dict() for o in v.optimizer] for k, v in self.module_topology.items() if v.optimizer is not None}
        state_dicts = {f"{priv[k]}_state_dict": v.state_dict() for k, v in self.module_topology.items()}
        state.update(optimizers)
        state.update(state_dicts)
        state.update(self.state)
        save_checkpoint(state, filename)
        print("Saved model to:", filename)

    def load_model(self, path, txt_file=None):
        """
        Load models main components from checkpoint : modules state-dictionaries & optimizers-state-dictionaries
        Method returns the checkpoint for any subclass of ARCH to load additional (model specific) components.
        :param path: file path
        :type path: Path | str
        :param txt_file: logging file
        :type txt_file: IOFile
        :return: torch.checkpoint | None
        :rtype: Any
        """
        priv = self._mod_topo_private
        self._store_path = path
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self._epochix = checkpoint['epoch_index']
            self.best_prec1 = checkpoint['best_prec1']
            [v.load_state_dict(checkpoint[f"{priv[k]}_state_dict"]) for k, v in self.module_topology.items()
             if f"{priv[k]}_state_dict" in checkpoint]
            [v.load_optimizer_state_dict(checkpoint[f"{priv[k]}_optim"]) for k, v in self.module_topology.items()
             if f"{k}_optim" in checkpoint]

            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']))
            if txt_file:
                print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']), file=txt_file)
            return checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(path))
            if txt_file:
                print("=> no checkpoint found at '{}'".format(path), file=self.p)
            return None

    def load_config(self, path):
        self.root_cfg = load_config(path)
        return self.root_cfg

    def override_child_cfg(self, _config: EasyDict | dict, module_id: str):
        """
        Overrides ARCH.Child module configuration based on root config values.
        Important:
        The mappings should be nested in the root config in the module's YAML objects (e.g. BACKBONE, DT...)
        and should not be nested in the child config!
        The keys in child config and root config must match (obviously).

        :param _config: child config
        :param module_id: child module ID
        :return: child cfg: EasyDict | dict
        """
        if module_id not in self.root_cfg.keys():
            raise KeyError("[CFG_OVERRIDE] Module ID not found in root cfg!")
        return config_exchange(_config, self.root_cfg[module_id])

    def get_loss(self, module_id: str = None):
        if module_id is None:
            return self.module_topology[self.root_cfg.TRACK_LOSS].loss
        if module_id not in self.module_topology.keys():
            raise KeyError(f"Module {module_id} not found in architecture topology")
        return self.module_topology[module_id].loss

    def calculate_loss(self, gt, pred):
        """
        Calculate loss (without grad!)
        """
        return self.module_topology[self.root_cfg.TRACK_LOSS].calculate_loss(gt, pred)

    def calculate_accuracy(self, output, target, topk=(1,)):
        prec = accuracy(F.softmax(output, dim=1), target, topk=topk)
        self.best_prec1 = prec[0] if prec[0] > self.best_prec1 else self.best_prec1
        return prec

    def get_epoch(self):
        return self._epochix

    def incr_epoch(self):
        self._epochix += 1

    def get_criterion(self):
        return self.module_topology[self.root_cfg.TRACK_CRITERION].criterion

    def init_weights(self):
        for _id, module in self.module_topology.items():
            init_weights(module, self.root_cfg[_id].INIT_WEIGHTS) if "INIT_WEIGHTS" in self.root_cfg[_id] else False

    def adjust_learning_rate(self, epoch_num):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        for k, mod in self.module_topology.items():
            if mod.training and mod.require_grad:
                mod.adjust_learning_rate(epoch=epoch_num)
