from enum import Enum, EnumMeta
from dataclasses import dataclass
from torch import nn


class ArchM(nn.Module):
    """
    Architecture Module - wraps around nn.Module and organizes necessary functions/values
    """
    class BaseConfig:
        __doc__ = "Base config class for yaml files. Override __doc__ for implementations."

    class ActivationFuncs(Enum):
        Relu = nn.ReLU
        Lrelu = nn.LeakyReLU
        Sigmoid = nn.Sigmoid

    class NormalizationFuncs(Enum):
        BatchNorm2d = nn.BatchNorm2d
        InstanceNorm2d = nn.InstanceNorm2d

    class PoolingFuncs(Enum):
        MaxPool2d = nn.MaxPool2d
        AveragePool2d = nn.AvgPool2d
        LPPool2d = nn.LPPool2d

    @staticmethod
    def get_func(fset: EnumMeta, name: str):
        return fset[name].value
