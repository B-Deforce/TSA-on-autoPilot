import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.model_utils import create_layers, get_loss_function


class CNN_AE_BASE(pl.LightningModule):
    """
    A vanilla convolutional autoencoder for 1D data (here, time series).
    """

    def __init__(self, encoder_params, decoder_params, lr, loss_type="mse"):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = create_layers(encoder_params)
        self.decoder = create_layers(decoder_params)
        self.lr = lr
        self.loss_fn = get_loss_function(loss_type)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
