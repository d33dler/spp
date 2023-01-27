from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

from models.architectures.classifier import DataHolder


class Encoder(nn.Module):
    def __init__(self, num_classes, norm_layer, use_bias):
        super().__init__()
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()
        # encoder
        self.encoder_cls = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(16),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1, bias=use_bias),
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

    def episode_forward(self, x: DataHolder):
        # Extract the encoder part of the autoencoder
        C, h, w = x.q.size()
        encoder_cls = self.encoder_cls
        encoder_conv = self.encoder_conv

        # Use the encoder to transform the input data
        x.q_enc = encoder_conv(x.q)
        x.q_sm = encoder_cls(x.q)
        S_enc = []
        for s in x.S:
            S_enc.append(encoder_conv(s))
        x.S_enc = S_enc
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

