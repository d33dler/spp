from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

from models.architectures.classifier import DataHolder
from models.utilities.knn_utils import get_norm_layer


class Encoder(nn.Module):
    def __init__(self, data: DataHolder):
        super().__init__()
        self.data = data
        num_classes = data.num_classes
        norm_layer, use_bias = get_norm_layer(data.cfg.ENCODER.NORM)
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()
        # encoder
        self.encoder_smax = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Softmax(dim=1)
        )

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),  # 32 x 21 x 21
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(16),
            nn.LeakyReLU(0.2, True),  # 16 x 21 x 21
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(8),
            nn.LeakyReLU(0.2, True),  # 8 x 21 x 21
            nn.MaxPool2d(kernel_size=3, stride=3)  # 8 x 7 x 7
        )

    def forward(self):
        # Extract the encoder part of the autoencoder
        data = self.data
        B, C, h, w = self.data.q.size()
        encoder_sm = self.encoder_smax
        encoder_conv = self.encoder_conv

        # Use the encoder to transform the input data
        data.q_enc = encoder_conv(data.q)
        data.q_sm = encoder_sm(data.q)
        if self.training:
            data.loss_q_sm = self.criterion(data.q_sm, data.targets)
        S_enc = []
        for s in data.S:
            S_enc.append(encoder_conv(s))
        data.S_enc = S_enc
        return data

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

