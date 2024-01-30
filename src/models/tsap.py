import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import higher

import src.custom_losses as c_loss
from src.models.detector_model import DetectorModel
from src.models.aug_model import AugmentationModel


class TSAP(pl.LightningModule):
    def __init__(
        self,
        aug_model_ckpt,
        detector_params,
        emb_size,
        aug_params,
        ts_input_size,
        a_init,
        aug_lr=0.001,
        det_lr=0.002,
        contam_rate=0.15,
        num_inner_loop=5,
        n_warm_start_epochs=3,
        normalize_emb=True,
        normalize_hparams=True,
    ):
        """
        TSAP model
        :param aug_model_ckpt: path to the augmentation model checkpoint
        :param detector_params: the architecture of the detector model
        :param emb_size: the number of samples to use for the validation loss
        :param aug_params: the augmentation hyperparameters to learn
        :param ts_input_size: the input size of the time series
        :param a_init: the initial values of the augmentation hyperparameters
        :param aug_lr: the learning rate of the augmentation model
        :param det_lr: the learning rate of the detector model
        :param contam_rate: the mixing rate used in phase (ii) as described in the paper
        :param num_inner_loop: the number of inner loop iterations
        :param n_warm_start_epochs: the number of epochs to warm start the detector model
        :param normalize_emb: whether to normalize the embeddings
        :param normalize_hparams: whether to normalize the augmentation hyperparameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.contam_rate = contam_rate
        self.ts_input_size = ts_input_size
        self.emb_size = emb_size
        self.normalize_hparams = normalize_hparams
        self.aug_params = self._process_augparams(aug_params)
        self.automatic_optimization = False  # turns off PL's automatic optimization
        self.num_inner_loop = num_inner_loop
        self.n_warm_start_epochs = n_warm_start_epochs
        self.det_lr = det_lr
        self.aug_lr = aug_lr
        self.normalize_emb = normalize_emb
        if self.normalize_emb:
            self.emb_normalizer = c_loss.EmbNormalizer()
        # init sub models
        self.detector_model = DetectorModel(
            detector_params=detector_params["params"],
            ts_input_size=self.ts_input_size,
        )
        self.augmentation = AugmentationModel(
            aug_model_ckpt,
            a_init=a_init,
            randomize=self.aug_params,
        )

        # init BCE loss and validation loss
        self.compute_det_loss = nn.BCEWithLogitsLoss()
        self.compute_val_loss = c_loss.WassersteinLoss(p=2, blur=0.05, scaling=0.8)
        # self.compute_val_loss = c_loss.MeanLoss(metric="euclidean")

        # init logging metrics
        self.roc_auc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        x_aug = self.augmentation(x)
        out = self.detector_model(x)
        return out, x_aug

    def training_step(self, batch, batch_idx):
        D_opt, A_opt = self.optimizers()
        with higher.innerloop_ctx(self.detector_model, D_opt.optimizer) as (
            fmodel,
            diffopt,
        ):
            Dloss = self._inner_loop_train(batch, fmodel, diffopt)
            Vloss, triple = self._outer_loop_train(fmodel, A_opt)

        with torch.no_grad():
            self._log_train_metrics(triple, Dloss, Vloss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1, 1).cpu()
        self._log_val_metrics(x, y)

    def _detector_step(self, batch, model):
        x_aug, x_trn, y_aug, y_trn = self._split_augment_batch(batch, outer_loop=False)
        x = torch.cat((x_aug, x_trn), dim=0)
        y = torch.cat((y_aug, y_trn), dim=0)
        y_pred = model(x)
        Dloss = self.compute_det_loss(y_pred, y)
        return Dloss

    def _inner_loop_train(self, batch, model, optimizer):
        model.train()
        Dloss = self._detector_step(batch, model)
        optimizer.step(Dloss)
        if self.num_inner_loop > 1:
            for _ in range(self.num_inner_loop):
                batch = self._get_batch()
                Dloss = self._detector_step(batch, model)
                optimizer.step(Dloss)
        return Dloss

    def _outer_loop_train(self, model, optimizer):
        model.eval()
        triple = self._get_embeddings(
            model.get_embed, self.emb_size
        )  # keep original triple for logging
        triple_scaled = self.emb_normalizer(*triple) if self.normalize_emb else triple
        Vloss = self.compute_val_loss(*triple_scaled)
        optimizer.zero_grad()
        self.manual_backward(Vloss)
        optimizer.step()
        self.detector_model.load_state_dict(model.state_dict())
        return Vloss, triple

    def _split_augment_batch(self, batch, outer_loop=False):
        x_raw, _ = batch
        batch_size = x_raw.size(0)

        # randomly select subset of training data to replace with augmented data
        cr = self.contam_rate if outer_loop else 0.5
        num_contam = round(cr * batch_size)
        idx = torch.randperm(batch_size)
        x_sub = x_raw[idx[:num_contam], :, :]
        x_trn = x_raw[idx[num_contam:], :, :]

        y_aug = torch.ones((num_contam, 1), device=self.device)  # initiate y with ones
        y_trn = torch.zeros(
            (batch_size - num_contam, 1), device=self.device
        )  # initiate y with zeros
        if outer_loop:
            with torch.enable_grad():
                x_aug = self.augmentation(x_sub)
        else:
            with torch.no_grad():
                x_aug = self.augmentation(x_sub)
        return x_aug, x_trn, y_aug, y_trn

    def _get_embeddings(self, model, size):
        x_trn_batches = []
        x_aug_batches = []
        x_val_batches = []
        accumulated_size = 0
        while accumulated_size < size:
            batch = self._get_batch()
            x_val, _ = self._get_xval_batch()

            # Determine the size for the current iteration
            current_batch_size = min(len(batch[0]), len(x_val), size - accumulated_size)

            # Truncate the current batch and x_val to the current_batch_size
            batch = batch[0][:current_batch_size], batch[1][:current_batch_size]
            x_val = x_val[:current_batch_size]

            # Split and augment the batch
            x_aug, x_trn, _, __ = self._split_augment_batch(batch, outer_loop=True)

            # Accumulate the batches
            x_trn_batches.append(x_trn)
            x_aug_batches.append(x_aug)
            x_val_batches.append(x_val)

            # Update the accumulated size
            accumulated_size += current_batch_size
        x_trn = torch.cat(x_trn_batches, dim=0)
        x_aug = torch.cat(x_aug_batches, dim=0)
        x_val = torch.cat(x_val_batches, dim=0)
        z_trn = model(x_trn).squeeze()
        z_aug = model(x_aug).squeeze()
        z_val = model(x_val).squeeze()
        return z_trn, z_aug, z_val

    def _get_xval_batch(self):
        try:
            batch = next(self.xval_iterator)
            batch = [i.to(self.device) for i in batch]
        except StopIteration:
            self.xval_iterator = iter(self.trainer.val_dataloaders)
            batch = next(self.xval_iterator)
            batch = [i.to(self.device) for i in batch]
        return batch

    def _get_batch(self):
        try:
            batch = next(self.batch_iterator)
            batch = [i.to(self.device) for i in batch]
        except StopIteration:
            self.batch_iterator = iter(self.trainer.train_dataloader)
            batch = next(self.batch_iterator)
            batch = [i.to(self.device) for i in batch]
        return batch

    def warm_start(self):
        """
        Warm start of detector
        """
        det_opt = self.optimizers()[0]
        batches_per_epoch = len(
            self.trainer.train_dataloader
        )  # The number of batches in one epoch

        batch_count = 0
        for _ in range(self.n_warm_start_epochs):  # Loop over epochs
            for __ in range(batches_per_epoch):  # Loop over batches within an epoch
                batch = self._get_batch()
                if batch_count % 50 == 0:
                    print("Warm start batch: ", batch_count)
                Dloss = self._detector_step(batch, self.detector_model)
                # backward pass
                det_opt.zero_grad()
                self.manual_backward(Dloss)
                det_opt.step()
                batch_count += 1

    def _process_augparams(self, cfg):
        aug_params = []
        for v in cfg.values():
            if isinstance(v, list):
                assert len(v) == 2, "hparam should have a start and stop"
                start = v[0] / self.ts_input_size if self.normalize_hparams else v[0]
                stop = v[1] / self.ts_input_size if self.normalize_hparams else v[1]
                aug_params.append((start, stop))
            elif v == "learn":
                aug_params.append(v)
            else:
                raise ValueError(
                    f"values in cfg should be one of list[int, int] or 'learn' but {v} was given"
                )
        return aug_params

    @torch.no_grad()
    def _log_train_metrics(self, triple, Dloss, Vloss):
        z_trn, z_aug, z_val = triple
        metrics = {}
        y = torch.tensor(
            data=[0] * z_trn.shape[0] + [1] * z_aug.shape[0],
            dtype=torch.int64,
        ).cpu()
        z_combined = torch.cat((z_trn, z_aug), dim=0)
        y_pred = self.detector_model.fc(z_combined)
        y_pred = torch.sigmoid(y_pred).cpu()
        total_loss = Dloss.cpu() + Vloss.cpu()
        metrics["det_train_roc_auc"] = self.roc_auc(y_pred, y)
        metrics["total_loss"] = total_loss
        metrics["det_train_loss"] = Dloss
        metrics["zval_loss"] = Vloss
        metrics["z_train_mean"] = torch.mean(z_trn).cpu().numpy().item()
        metrics["z_aug_mean"] = torch.mean(z_aug).cpu().numpy().item()
        metrics["z_val_mean"] = torch.mean(z_val).cpu().numpy().item()
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=True
                    if metric_name in ["total_loss", "det_train_loss", "zval_loss"]
                    else False,
                )
        for i, param in enumerate(self.augmentation.params_sample):
            # Assuming each param is a 1D tensor and you want to log its individual elements
            self.log(f"a_hp_{i}", param, prog_bar=True)

    @torch.no_grad()
    def _log_val_metrics(self, x, y):
        z_val = self.detector_model.get_embed(x)
        y_pred = self.detector_model.fc(z_val).cpu()
        y_pred = torch.sigmoid(y_pred)
        metrics = {}
        metrics["det_val_roc_auc"] = self.roc_auc(y_pred, y)
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                self.log(metric_name, metric_value)

    ############
    # PL hooks #
    ############

    def training_epoch_start(self, batch, batch_idx):
        """
        pytorch-lightning hook
        """
        self.batch_iterator = iter(self.trainer.train_dataloader)
        self.xval_iterator = iter(self.trainer.val_dataloaders)

    def on_train_start(self):
        """
        pytorch-lightning hook
        """
        self.batch_iterator = iter(self.trainer.train_dataloader)
        self.batch_size = self.trainer.train_dataloader.batch_size
        self.xval_iterator = iter(self.trainer.val_dataloaders)
        # Call the dummy training step at the beginning of training
        if self.n_warm_start_epochs > 0:
            print("Warm starting detector...")
            self.warm_start()
            print("Warm start complete.")

    def configure_optimizers(self):
        """
        pytorch-lightning hook
        """
        opt_det = torch.optim.Adam(self.detector_model.parameters(), lr=self.det_lr)
        opt_aug = torch.optim.Adam(
            [
                self.augmentation.learnable_params[i]
                for i, v in enumerate(self.aug_params)
                if v == "learn"
            ],
            lr=self.aug_lr,
        )
        return [
            {"optimizer": opt_det},
            {"optimizer": opt_aug},
        ]
