import pytorch_lightning as pl
import torch
from torch import nn
import warnings
import yaml
import numpy as np
from src.models.cond_cnn import COND_CNN_AE


class AugmentationModel(pl.LightningModule):
    def __init__(
        self,
        aug_model_ckpt,
        a_init=None,
        seed=96,
        randomize=None,
    ):
        """
        Wrapper that takes a trained augmentation model and uses it to augment the input data with learnable hyperparameters
        :param augmentation_model_checkpoint: the checkpoint of the augmentation model to use
        :param meta_data_dim: the dimension of the meta data
        :param fix_augmentation_model: whether to fix the augmentation model's parameters
        :param a_init: the initial value of the augmentation parameters. If None, the parameters will be initialized randomly
        of size meta_data_dim. Else, the parameters will be initialized to a_init (rendering meta_data_dim irrelevant)
        :param randomize: a list of tuples (idx, dtype,  (start, stop)) of indices of the augmentation parameters to randomize
        """
        super().__init__()
        self.save_hyperparameters()
        with open("configs/f_augment/a=3.yml") as cfg:
            m_config = yaml.safe_load(cfg)
            m_config = m_config["model_params"]
        model = COND_CNN_AE(
            ts_input_size=m_config.get("ts_input_size"),
            meta_hidden_size=m_config.get("meta_hidden_size"),
            n_meta_layers=m_config.get("n_meta_layers"),
            meta_input_size=m_config.get("meta_input_size"),
            encoder_params=m_config.get("encoder_params"),
            decoder_params=m_config.get("decoder_params"),
            lr=m_config.get("lr"),
        )
        self.augmentation = model.load_from_checkpoint(
            aug_model_ckpt, map_location=self.device
        )
        for param in self.augmentation.parameters():
            param.requires_grad = False
        self.a_init = a_init
        self.a_size = len(a_init)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.param_sample = None
        self.learnable_params = nn.ParameterList()
        self.batch_size = 10

        # Initialize parameters
        for i, v in enumerate(self.randomize):
            if v == "learn":
                # Learnable parameter with gradients
                param = nn.Parameter(
                    torch.tensor(self.a_init[i], device=self.device), requires_grad=True
                )
            else:
                # Non-learnable parameter without gradients
                start, stop = v
                param = nn.Parameter(
                    torch.tensor(
                        self.rng.uniform(start, stop, size=(self.batch_size, 1)),
                        device=self.device,
                    ),
                    requires_grad=False,
                )
            self.learnable_params.append(param)

    def randomize_params(self, batch_size):
        # Update non-learnable parameters for the batch
        for i, v in enumerate(self.randomize):
            if v != "learn":
                start, stop = v
                random_tensor = torch.tensor(
                    self.rng.uniform(
                        start,
                        stop,
                        size=batch_size,
                    ),
                    device=self.device,
                    dtype=torch.float,
                ).unsqueeze(1)
                self.learnable_params[i] = nn.Parameter(
                    random_tensor, requires_grad=False
                )

    def forward(self, x):
        self.batch_size = x.size(0)
        # Randomize non-learnable params
        self.randomize_params(self.batch_size)

        # Process and concatenate parameters
        broadcasted_params = []
        for i, param in enumerate(self.learnable_params):
            if self.randomize[i] == "learn":
                # Expand learnable parameters
                expanded_param = param.unsqueeze(0).expand(self.batch_size, -1)
                broadcasted_params.append(expanded_param)
            else:
                # Non-learnable parameters are already the correct shape
                broadcasted_params.append(param)

        concatenated_params = torch.cat(broadcasted_params, dim=1)

        with torch.no_grad():
            self.params_sample = concatenated_params[0].cpu().tolist()

        # Apply the augmentation or any other operation involving the parameters
        x_aug, _ = self.augmentation(x, concatenated_params)

        return x_aug

    def on_save_checkpoint(self, checkpoint):
        # Update the checkpoint with the current hyperparameters
        checkpoint[
            "batch_size"
        ] = self.batch_size  # save last known batch size, needed when loading model
