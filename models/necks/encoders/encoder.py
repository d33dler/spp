from typing import Any, OrderedDict

import torch
import torch.nn as nn
from torch import optim, Tensor

from models.clustering import KNN_itc
from models.interfaces.arch_module import ARCH
from models.utilities.utils import DataHolder, get_norm_layer, init_weights


class Encoder(ARCH.Child):
    def __init__(self, data: DataHolder):
        super().__init__()

        self.data = data
        num_classes = data.num_classes
        norm_layer, use_bias = get_norm_layer(data.cfg.ENCODER.NORM)

        # FC blocks + log-softmax output
        self.encoder_smax = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data.cfg.ENCODER.FLATTENED_INPUT, 512, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes, bias=use_bias),
            nn.LogSoftmax(dim=1)
        )
        # Convolutional blocks for dimensionality reduction
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(data.cfg.ENCODER.INPUT_SIZE, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),  # 32 x 25 x 25
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(16),
            nn.LeakyReLU(0.2, True),  # 16 x 25 x 25
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(8),
            nn.LeakyReLU(0.2, True),  # 8 x 25 x 25
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8 x 8 x 8
        )

        cfg = data.cfg.ENCODER
        self.lr = [cfg.SMAX_LR, cfg.RED_LR]
        init_weights(self.encoder_conv, cfg.INIT_WEIGHTS)
        init_weights(self.encoder_smax, cfg.INIT_WEIGHTS)
        self.optimizer = [
            optim.Adam(self.encoder_smax.parameters(), lr=cfg.SMAX_LR, betas=tuple(cfg.BETA_ONE)),
            optim.Adam(self.encoder_conv.parameters(), lr=cfg.RED_LR, betas=tuple(cfg.BETA_ONE))]
        self.criterion = [nn.CrossEntropyLoss().cuda(), nn.CrossEntropyLoss().cuda()]
        self.knn = KNN_itc(data.k_neighbors)

    def forward(self):
        # Extract the encoder part of the encoders
        data = self.data
        B, C, h, w = self.data.q.size()
        encoder_sm = self.encoder_smax
        encoder_conv = self.encoder_conv
        # Use the encoder to transform the input data
        data.q_reduced = encoder_conv(data.q)

        data.q_smax = encoder_sm(data.q)
        S_red = []
        data.S_reduced = S_red
        for s in data.DLD_topk:
            support_set_sam = encoder_conv(s)
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().reshape((C, -1))
            data.S.append(support_set_sam)
            S_red.append(support_set_sam)
        data.sim_list_REDUCED, data.DLD_topk = self.knn.forward(data.q_reduced, data.S_reduced)

        if self.training:
            self.backward([data.q_smax, data.sim_list_REDUCED], data.targets)
        return data

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def state_dict(self, destination = None, prefix: str = '', keep_vars: bool = False) :
        return {
            'ENCODER_CONV': self.encoder_conv.state_dict(destination, prefix, keep_vars),
            'ENCODER_SMAX': self.encoder_smax.state_dict(destination, prefix, keep_vars)
        }

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        self.encoder_conv.load_state_dict(state_dict['ENCODER_CONV'], strict)
        self.encoder_smax.load_state_dict(state_dict['ENCODER_SMAX'], strict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
