import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from torch import nn
from yamldataclassconfig import YamlDataClassConfig, create_file_path_field
from dataclasses_json import DataClassJsonMixin
from models.interfaces.arch_module import ArchM


class BaseBackbone2d(ArchM.Child):
    """
    Abstraction class for 2d backbones implementing using torch
    """
    features: nn.Sequential
    norm_layers: List[int] = []
    ACTIVATION_F: ArchM.ActivationFuncs
    NORMALIZATION_F: ArchM.NormalizationFuncs
    POOLING_F: ArchM.PoolingFuncs

    @dataclass
    class _CFG(ArchM.BaseConfig):

        INP_CHANNELS: List[int] = field(default_factory=list)
        OUT_CHANNELS: List[int] = field(default_factory=list)
        LAYER_STRIDES: List[int] = field(default_factory=list)
        KERNEL_SIZES: List[int] = field(default_factory=list)
        LAYER_POOLS: List[int] = field(default_factory=list)
        LAYER_PADDINGS: List[int] = field(default_factory=list)
        NUM_FILTERS: List[int] = field(default_factory=list)
        MOMENTUM: List[float] = field(default_factory=list)

        UPSAMPLE_STRIDES: List[int] = field(default_factory=list)  # []
        NUM_UPSAMPLE_FILTERS: List[int] = field(default_factory=list)  # []

        ACTIVATION: str = 'Relu'
        NORMALIZATION: str = 'BatchNorm2d'
        POOLING: str = 'MaxPool2d'

        NORM_ARGS: dict = field(default_factory=dict)  # for now only one set of parameters
        POOL_ARGS: dict = field(default_factory=dict)
        DEBLOCK_ARGS: dict = field(default_factory=dict)
        FILE_TYPE: str = "YAML"

        def load_cfg(self, *args, **kwargs):
            raise NotImplementedError("Using base class load(). Must call() implementation instance's function.")

    class RemoteYamlConfig(_CFG, YamlDataClassConfig, ABC):
        pass

    class RemoteJsonConfig(_CFG, DataClassJsonMixin, ABC):
        pass

    def _YamlCFG(self, config: RemoteYamlConfig, __file__):
        """Yaml mapping config object class using YamlDataClassConfig."""
        config.FILE_PATH = create_file_path_field(Path(
            __file__).parent / 'config.yaml')  # os.path.join(os.path.dirname(os.path.realpath(__file__)), Path(load_cfg))
        print(Path(__file__).parent / 'config.yaml')
        config.load(Path(__file__).parent / 'config.yaml')  # check!
        return config

    def _JsonCFG(self, config: RemoteJsonConfig, __file__):
        """Yaml mapping config object class using YamlDataClassConfig."""
        config.FILE_PATH = create_file_path_field(Path(__file__).parent.parent / 'config.json')
        with open(config.FILE_PATH, 'r') as f:
            return self.from_json(json.load(f))

    cfg: Union[RemoteJsonConfig, RemoteYamlConfig]
    blocks: nn.ModuleList
    deblocks: nn.ModuleList

    def collect_funcs(self):
        self.ACTIVATION_F, \
            self.NORMALIZATION_F, \
            self.POOLING_F = \
            [ArchM.get_func(fset, name)
             for fset, name in [(ArchM.ActivationFuncs, self.cfg.ACTIVATION),
                                (ArchM.NormalizationFuncs, self.cfg.NORMALIZATION),
                                (ArchM.PoolingFuncs, self.cfg.POOLING)]]

    def __init__(self, remote_cfg: Union[RemoteJsonConfig, RemoteYamlConfig]):  # remove CFG and refer to self
        """

        :param config:
        :type config:
        :param args:
        :type args:
        """
        super().__init__()

        config: Union[BaseBackbone2d._YamlCFG, BaseBackbone2d._JsonCFG]  # TODO refactor this
        if remote_cfg.FILE_TYPE == "JSON":
            self.cfg = self._JsonCFG(remote_cfg, remote_cfg.FILE_PATH)
        elif remote_cfg.FILE_TYPE == "YAML":
            self.cfg = self._YamlCFG(remote_cfg, remote_cfg.FILE_PATH)
        else:
            raise AttributeError("Config file type not supported")
        self.collect_funcs()
        use_bias = self.ACTIVATION_F == nn.InstanceNorm2d
        # TODO finish abstraction

        # assert len(self.cfg.LAYER_NUMS) == len(self.cfg.LAYER_STRIDES) == len(self.cfg.NUM_FILTERS)
        # layer_nums = self.cfg.LAYER_NUMS
        # layer_strides = self.cfg.LAYER_STRIDES
        # num_filters = self.cfg.NUM_FILTERS
        #
        # if self.cfg.UPSAMPLE_STRIDES is not None:
        #     assert len(self.cfg.UPSAMPLE_STRIDES) == len(self.cfg.NUM_UPSAMPLE_FILTERS)
        #     num_upsample_filters = self.cfg.NUM_UPSAMPLE_FILTERS
        #     upsample_strides = self.cfg.UPSAMPLE_STRIDES
        # else:
        #     upsample_strides = num_upsample_filters = []
        #
        # num_levels = len(layer_nums)
        # c_in_list = [self.cfg.INPUT_CHANNELS, *num_filters[:-1]]
        # self.blocks = nn.ModuleList()
        # self.deblocks = nn.ModuleList()
        #
        # # nn.ZeroPad2d(1) TODO look at this
        #
        # for lvl_i in range(num_levels):
        #     cur_layers = []
        #     for lr_ix in range(layer_nums[lvl_i]):
        #         self._conv_layer(cur_layers, lvl_i)
        #         self._norm_layer(cur_layers, lvl_i)
        #         self._activation_layer(cur_layers, lvl_i)
        #         self._pooling_layer(cur_layers, lvl_i, lr_ix)
        #
        #     self.blocks.append(nn.Sequential(*cur_layers))
        #     if len(upsample_strides) > 0:
        #         stride = upsample_strides[lvl_i]
        #         if stride >= 1:
        #             self.deblocks.append(nn.Sequential(
        #                 nn.ConvTranspose2d(
        #                     num_filters[lvl_i], num_upsample_filters[lvl_i],
        #                     upsample_strides[lvl_i],
        #                     stride=upsample_strides[lvl_i], bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[lvl_i], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #         else:
        #             stride = np.round(1 / stride).astype(np.int)
        #             self.deblocks.append(nn.Sequential(
        #                 nn.Conv2d(
        #                     num_filters[lvl_i], num_upsample_filters[lvl_i],
        #                     stride,
        #                     stride=stride, bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[lvl_i], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #
        # c_in = sum(num_upsample_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))
        #
        # self.num_bev_features = c_in

    def build(self):
        pass

    def _conv_layer(self, lvl: List[nn.Module], ix):
        lvl.extend(nn.Conv2d(self.cfg.NUM_FILTERS[ix], self.cfg.NUM_FILTERS[ix], kernel_size=3, padding=1, bias=False))

    def _norm_layer(self, lvl: List[nn.Module], ix):
        if self.cfg.NORMALIZATION_F is not None:
            lvl.extend(self.cfg.NORMALIZATION_F(self.cfg.NUM_FILTERS[ix], **self.cfg.NORM_ARGS[ix]))

    def _activation_layer(self, lvl: List[nn.Module], ix):
        if self.cfg.ACTIVATION_F is not None:
            lvl.extend(self.cfg.ACTIVATION_F(self.cfg.NUM_FILTERS[ix]))
        raise ValueError("Missing activation function specification 'ACTIVATION' in config file!")

    def _pooling_layer(self, lvl: List[nn.Module], blk_ix, layer_ix):
        if self.cfg.POOLING is not None:
            try:
                lvl.extend(self.cfg.POOLING_F(**self.cfg.POOL_ARGS[blk_ix][layer_ix]))
            except KeyError:
                return

    def forward(self, *args):
        raise NotImplementedError("forward() method not implemented")

    def freeze_layers(self):
        for layer_ix in self.norm_layers:
            self.features[layer_ix] = self.features[layer_ix].requires_grad_(False)
