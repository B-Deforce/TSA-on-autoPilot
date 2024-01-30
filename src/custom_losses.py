import torch
import math
import geomloss
import sklearn.metrics
import numpy as np


def distance(x, y, metric="euclidean", aggregation="mean"):
    if metric == "euclidean":
        out = torch.cdist(x, y, p=2)
    else:
        raise ValueError(metric)

    if aggregation == "mean":
        return out.mean()
    if aggregation == "sum":
        return out.sum()
    elif aggregation == "none":
        return out
    else:
        raise ValueError(aggregation)


def roc_auc_score(y_true, y_pred):
    return sklearn.metrics.roc_auc_score(y_true, y_pred)


def aupr(y_true, y_pred):
    return sklearn.metrics.average_precision_score(y_true, y_pred)


def get_best_f1(label, score):
    precision, recall, ths = sklearn.metrics.precision_recall_curve(
        y_true=label, probas_pred=score
    )
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    best_th = ths[np.argmax(f1)]
    return best_f1, best_p, best_r, best_th


class EmbNormalizer:
    def __init__(self, mode="tpsd"):
        self.mode = mode
        self.std = None
        self.emb_norm = None
        self.emb_size = None
        self.emb_mean = None

    def __call__(self, emb_x, emb_y, emb_z):
        if self.mode == "tpsd":
            emb_all = torch.cat([emb_x, emb_y, emb_z])
            mean = emb_all.mean(0)
            std = torch.norm(emb_all - mean) / math.sqrt(emb_all.size(0))
            emb_x = (emb_x - mean) / std
            emb_y = (emb_y - mean) / std
            emb_z = (emb_z - mean) / std
            return emb_x, emb_y, emb_z
        else:
            raise ValueError(self.mode)


class MeanLoss:
    def __init__(self, metric, aggregation="mean"):
        self.metric = metric
        self.aggregation = aggregation

    def __call__(self, emb_x, emb_y, emb_z):
        mean_x = emb_x.mean(0, keepdim=True)
        mean_y = emb_y.mean(0, keepdim=True)
        d1 = distance(mean_x, emb_z, self.metric, self.aggregation)
        d2 = distance(mean_y, emb_z, self.metric, self.aggregation)
        return (d1 + d2) / 2


class WassersteinLoss:
    """
    Approximate Wasserstein distance between two sets of samples using Sinkhorn algorithm.
    """

    def __init__(self, p=2, blur=0.05, scaling=0.9, diameter=None):
        self.p = p
        self.blur = blur
        self.scaling = scaling
        self.W_loss = geomloss.SamplesLoss(
            "sinkhorn", p=self.p, blur=self.blur, scaling=self.scaling
        )

    def __call__(self, emb_x, emb_y, emb_z):
        emb_xy = torch.cat((emb_x, emb_y), dim=0)
        loss = self.W_loss(emb_xy, emb_z)
        return loss
