from torch import nn
import torch
import torch.nn.functional as F

from src.models.base_cnn_ae import CNN_AE_BASE
import pytorch_lightning as pl


class COND_CNN_AE(CNN_AE_BASE):
    def __init__(
        self,
        ts_input_size,
        meta_hidden_size,
        n_meta_layers,
        meta_input_size,
        encoder_params,
        decoder_params,
        lr,
    ):
        super().__init__(
            encoder_params=encoder_params, decoder_params=decoder_params, lr=lr
        )
        self.save_hyperparameters()

        self.n_meta_layers = n_meta_layers
        if self.n_meta_layers < 2:
            raise ValueError("The number of cond layers must be at least 2")
        self.ts_input_size = ts_input_size
        self.meta_hidden_size = meta_hidden_size
        self.meta_input_size = meta_input_size
        # Perform a dummy forward pass to calculate the encoder's output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.ts_input_size)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_size = dummy_output.size(2)
            self.encoder_hidden_size = dummy_output.size(1)

        layers = []
        for _ in range(self.n_meta_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(self.meta_input_size, self.meta_hidden_size),
                    nn.ReLU(),
                )
            )
        layers.append(
            nn.Linear(
                self.meta_hidden_size,
                self.encoder_hidden_size * self.encoder_output_size,
            )
        )  # Ensure the last layer outputs a tensor with size equal to the encoder's output
        self.nn_cond = nn.ModuleList(layers)

    def forward(self, x, meta):
        encoded = F.relu(self.encoder(x))

        meta_processed = meta
        for layer in self.nn_cond:
            meta_processed = layer(meta_processed)

        meta_processed = F.relu(meta_processed).reshape(
            -1, self.encoder_hidden_size, self.encoder_output_size
        )
        augmented = encoded + meta_processed

        augmented = self.decoder(augmented)
        decoded = self.decoder(encoded)

        return augmented, decoded

    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        z, xt = self(x, meta)
        aloss = F.mse_loss(z, y)
        rloss = F.mse_loss(x, xt)
        loss = (aloss + rloss) / 2
        self.log("train_loss", loss, prog_bar=True)
        self.log("aloss", aloss, prog_bar=True)
        self.log("rloss", rloss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        z, _ = self(x, meta)
        loss = F.mse_loss(z, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        z, _ = self(x, meta)
        loss = F.mse_loss(z, y)
        self.log("test_loss", loss)
        return loss

    def inject(self, x, meta):
        z, _ = self(x, meta)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
