import torch

from src.models.base_cnn_ae import CNN_AE_BASE
from src.models.cond_cnn import COND_CNN_AE
from src.models.tsap import TSAP


def get_model_from_ckpt(model_type, load_ckpt):
    model_mapper = {
        "BASE_CNN": CNN_AE_BASE,
        "COND_CNN": COND_CNN_AE,
        "TSAP": TSAP,
    }
    ModelClass = model_mapper[model_type]
    model = ModelClass.load_from_checkpoint(
        load_ckpt,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


def get_model(config, load_ckpt=None):
    model_type = config["model_type"].upper()
    if load_ckpt is not None:
        model = get_model_from_ckpt(model_type, load_ckpt)
        return model
    else:
        if model_type == "BASE_CNN":
            return CNN_AE_BASE(
                encoder_params=config["encoder_params"],
                decoder_params=config["decoder_params"],
                lr=config["lr"],
                loss_type=config["loss_type"],
            )
        elif model_type == "COND_CNN":
            return COND_CNN_AE(
                ts_input_size=config.get("ts_input_size"),
                meta_hidden_size=config.get("meta_hidden_size"),
                n_meta_layers=config.get("n_meta_layers"),
                meta_input_size=config.get("meta_input_size"),
                encoder_params=config.get("encoder_params"),
                decoder_params=config.get("decoder_params"),
                lr=config.get("lr"),
            )
        elif model_type == "TSAP":
            return TSAP(
                aug_model_ckpt=config.get("aug_model_ckpt"),
                detector_params=config.get("detector_params"),
                aug_params=config.get("aug_params"),
                emb_size=config.get("emb_size"),
                ts_input_size=config.get("ts_input_size"),
                contam_rate=config.get("contam_rate"),
                a_init=config.get("a_init"),
                num_inner_loop=config.get("num_inner_loop"),
                n_warm_start_epochs=config.get("n_warm_start_epochs"),
                det_lr=config.get("det_lr"),
                aug_lr=config.get("aug_lr"),
                normalize_hparams=config.get("normalize_hparams", True),
                normalize_emb=config.get("normalize_emb", True),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
