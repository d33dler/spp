from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

from models.architectures.classifier import DataHolder


class Encoder(pl.LightningModule):
    def __init__(self, norm_layer, use_bias):
        super().__init__()
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()
        # encoder
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * 21 * 21, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # bottleneck
        )
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(16),
            nn.LeakyReLU(0.2, True),
        )

    def episode_forward(self, x: DataHolder):
        # Extract the encoder part of the autoencoder
        C, h, w = x.q.size()
        encoder = self.encoder_lin
        # Use the encoder to transform the input data
        x.q_enc = encoder(x.q.reshape((C, -1)))
        S_enc = []
        for s in x.S:
            S_enc.append(encoder(s.reshape((C, -1))))
        x.S_enc = encoder(x)
        return x

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch  # We do not need the labels
        x_hat = self.encoder_lin(x)  # TODO reshaping?
        loss = self.criterion(x, x_hat, reduction="none")  # MSE
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

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

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
