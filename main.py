import os
import yaml
import pytorch_lightning as pl
import torch
import argparse
import datetime
import glob
import wandb
import numpy as np

# custom imports
from src.data.data_preprocessor import DataFetcher, DataSplitter, DataScaler
import src.custom_losses as c_loss

# from src.eval.evaluate import evaluate_model, get_predictions, save_results
from src.utils import setup_checkpoint, setup_logger_and_checkpoint
from src.data.custom_dataloader import get_dataloaders
from src.anomalies import mix_anomalies
from src.models.model_factory import get_model

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")

    # set seed
    pl.seed_everything(96)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    time_id = str(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--anom_data_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, required=False)
    parser.add_argument("--ckpt_monitor", type=str, required=False)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--load_from_ckpt", type=str, default=None)
    parser.add_argument(
        "--train_mode", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use wandb for logging",
    )
    args = parser.parse_args()

    # load configs
    with open(args.config_path) as cfg:
        config = yaml.safe_load(cfg)

    d_config = config["data_params"]
    m_config = config["model_params"]
    if args.anom_data_path is not None:
        d_config["data_paths"]["anom_data_path"] = args.anom_data_path

    # initialize fetcher, scaler, and splitter
    fetcher = DataFetcher(d_config["data_paths"])
    splitter = DataSplitter(d_config["data_splits"])
    scaler = DataScaler(
        d_config["data_scaling"]
    )  # data scaling is typically done before, so this is not really used
    # TODO: if scaler.config is not None:

    x, y, meta = fetcher.load_data()
    splitted_data = splitter.split_data(x, y, meta)
    print("Mixing anomalies...")
    splitted_data = mix_anomalies(m_config["mixing_params"], splitted_data)
    train, val, test = splitted_data
    print("train:", train[0].shape, train[1].shape)
    print("val:", val[0].shape, val[1].shape)
    print("test:", test[0].shape, test[1].shape)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train,
        val,
        test,
        batch_size=m_config["batch_size"],
    )

    # setup checkpoint and/or logger
    if (args.wandb) & (args.train_mode):
        logger, ckpt_callback = setup_logger_and_checkpoint(
            name=f"{args.ckpt_name}-{time_id}",
            project="TSA-AP",  # fixed logging repo on wandb
            monitor=args.ckpt_monitor,
        )
        logger.experiment.config.update(config)
    elif args.train_mode:
        ckpt_callback = setup_checkpoint(
            f"{args.ckpt_name}",
            monitor=args.ckpt_monitor,
        )

    # get some data info
    ts_size = x.shape[1]

    if args.train_mode:
        print("Training model...")
        # get model
        model = get_model(m_config)
        # fit model
        trainer = pl.Trainer(
            max_epochs=m_config["epochs"],
            callbacks=[ckpt_callback],
            logger=logger
            if args.wandb
            else False,  # if False, uses default TensorBoard logger
            num_sanity_val_steps=0,
        )
        print(f"Training {m_config['model_type']}...")
        trainer.fit(model, train_dataloader, val_dataloader)
        print(
            f"Finished training {m_config['model_type']}. Best checkpoint: {ckpt_callback.best_model_path}"
        )
    if args.test_mode:
        print("Testing model...")
        if m_config["model_type"].upper() not in ["IF", "LOF", "OCSVM"]:
            if args.load_from_ckpt is None:
                assert args.train_mode, "Please provide a checkpoint path"
                best_ckpt = ckpt_callback.best_model_path
            elif m_config["model_type"] not in ["IF"]:
                # Search for all .ckpt files in the specified directory
                ckpt_files = glob.glob(args.load_from_ckpt)
                best_ckpt = ckpt_files[0]
            print(f"Loading model from {best_ckpt}...")
            model = get_model(m_config, load_ckpt=best_ckpt)
            print("Model loaded.")
            model = model.eval().to("cpu")

        if "TSAP" in m_config["model_type"].upper():
            scores = []
            y_all = []
            for batch in test_dataloader:
                x, y = batch
                score = model.detector_model(x)
                scores.append(score)
                y_all.append(y)
            scores_all = torch.cat(scores, dim=0).detach().numpy().squeeze()
            y_all = torch.cat(y_all, dim=0).detach().numpy().squeeze()
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")

        elif "LSTM" in m_config["model_type"].upper():
            scores = []
            y_all = []
            for batch in test_dataloader:
                x, y = batch
                score = model.predict_step(x)
                scores.append(score["prediction"])
                y_all.append(y)
            scores_all = torch.cat(scores, dim=0).detach().numpy().squeeze()
            print(scores_all.shape)
            y_all = torch.cat(y_all, dim=0).detach().numpy().squeeze()
            print(y_all.shape)
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")
            # evaluate_lstm(model, train_dataloader, test_dataloader)

        elif "USAD" in m_config["model_type"].upper():
            # run through all batches in test_dataloader and get the scores
            scores = []
            y_all = []
            for batch in test_dataloader:
                x, y = batch
                score = model.score(x)
                scores.append(score)
                y_all.append(y)
            scores_all = torch.cat(scores, dim=0).detach().numpy().squeeze()
            y_all = torch.cat(y_all, dim=0).detach().numpy().squeeze()
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")

        elif "SRCNN" in m_config["model_type"].upper():
            # run through all batches in test_dataloader and get the scores
            scores = []
            y_all = []
            for batch in test_dataloader:
                x, y = batch
                x = model.spectral_residual(x)
                score = model(x)
                scores.append(score)
                y_all.append(y)
            scores_all = torch.cat(scores, dim=0).detach().numpy().squeeze()
            y_all = torch.cat(y_all, dim=0).detach().numpy().squeeze()
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")

        elif m_config["model_type"].upper() in ["IF"]:
            counter = 0
            y_all = []
            scores_all = []
            # ids = np.where(test[1] == 1.0)[0]
            model = get_model(m_config).fit(train[0])
            scores_all = -model.decision_function(test[0])
            y_all = test[1].squeeze()
            print(y_all)
            print(scores_all)
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")

        elif m_config["model_type"].upper() in ["LOF"]:
            counter = 0
            y_all = []
            scores_all = []
            # ids = np.where(test[1] == 1.0)[0]
            for batch in zip(test[0], test[1]):
                model = get_model(m_config)
                x, y = batch
                x = x.reshape(-1, 1)
                counter += 1
                model.fit_predict(x)
                scores = model.negative_outlier_factor_
                # negative == anomaly, pos == inlier
                min_score = scores.min()  # get min of window
                scores_all.append(min_score.item())
                y_all.append(y.item())
                if counter % 20 == 0:
                    print(
                        f"{counter + 1} done, {test[0].shape[0] - counter + 1} to go."
                    )
            y_all = np.array(y_all)
            scores_all = -np.array(scores_all)
            print(y_all)
            print(scores_all)
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")

        elif m_config["model_type"].upper() in ["OCSVM"]:
            counter = 0
            y_all = []
            scores_all = []
            # ids = np.where(test[1] == 1.0)[0]
            model = get_model(m_config).fit(train[0])
            scores_all = -model.decision_function(test[0])
            y_all = test[1].squeeze()
            print(y_all)
            print(scores_all)
            # calculate the AUC
            auc = c_loss.roc_auc_score(y_all, scores_all)
            print(f"AUC: {auc}")
            # get recall
            f1_metrics = c_loss.get_best_f1(y_all, scores_all)
            print(f"f1: {f1_metrics[0]}, p: {f1_metrics[1]}, r: {f1_metrics[2]}")
