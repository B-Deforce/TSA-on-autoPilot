import csv
import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import os


def setup_logger_and_checkpoint(name, project, monitor):
    logger = WandbLogger(project=project, name=name, log_model=True)
    model_checkpoint = setup_checkpoint(name, monitor)
    return logger, model_checkpoint


def setup_checkpoint(name, monitor):
    os.makedirs(f"checkpoints/{name}/", exist_ok=True)
    return pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=f"checkpoints/{name}/",
        filename=f"{name.split('/')[-1]}" + "-{epoch}" + "-{" + monitor + ":.2f}",
        save_top_k=2,
        save_last=True,
        mode="min",
    )


def write_to_csv(file_path, headers, values):
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(dict(zip(headers, values)))
