"""
General utilities
"""
from pathlib import Path

import yaml
from easydict import EasyDict


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
