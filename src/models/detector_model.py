from torch import nn
import torch
import torchmetrics
from src.models.model_utils import create_layers
import warnings


class DetectorModel(nn.Module):
    def __init__(self, detector_params, ts_input_size):
        super().__init__()
        self.ts_input_size = ts_input_size

        self.encoder = create_layers(detector_params)
        # Perform a dummy forward pass to calculate the encoder's output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.ts_input_size)
            dummy_output = self.encoder(dummy_input)
            print(f"Encoder output size: {dummy_output.size()}")
            self.n_channels = dummy_output.size(1)
        self.fc = nn.Linear(self.n_channels, 1)
        self.compute_det_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        embed = self.encoder(x)
        out = self.fc(embed)
        return out

    def get_embed(self, x):
        embed = self.encoder(x)
        return embed
