from torch import nn


def create_layers(layer_configs):
    layers = []
    for config in layer_configs:
        if config["type"] == "conv":
            layers.append(nn.Conv1d(**config["params"]))
        elif config["type"] == "conv_transpose":
            layers.append(nn.ConvTranspose1d(**config["params"]))
        elif config["type"] == "batch_norm":
            layers.append(nn.BatchNorm1d(**config["params"]))
        elif config["type"] == "relu":
            layers.append(nn.ReLU())
        elif config["type"] == "max_pool":
            layers.append(nn.MaxPool1d(**config["params"]))
        elif config["type"] == "dropout":
            layers.append(nn.Dropout(**config["params"]))
        elif config["type"] == "flatten":
            layers.append(nn.Flatten())
        elif config["type"] == "lstm":
            layers.append(nn.LSTM(**config["params"], batch_first=True))
        elif config["type"] == "linear":
            layers.append(nn.Linear(**config["params"]))
        elif config["type"] == "avg_pool":
            layers.append(nn.AvgPool1d(**config["params"]))
        else:
            raise ValueError(f"Unknown layer type: {config['type']}")
    return nn.Sequential(*layers)


def get_loss_function(loss_type):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
