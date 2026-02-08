from __future__ import annotations
import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class MetricResult:
    accuracy: float
    auroc: float
    auprc: float
    f1_macro: float

def compute_metrics(y_true, y_prob, y_pred, num_classes: int) -> MetricResult:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))

    auroc = np.nan
    auprc = np.nan
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
            auprc = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            y_true_1h = np.eye(num_classes)[y_true]
            auprc = float(average_precision_score(y_true_1h, y_prob, average="macro"))
    except Exception:
        pass

    return MetricResult(accuracy=acc, auroc=auroc, auprc=auprc, f1_macro=f1m)

def save_json(path: str, obj) -> None:
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
