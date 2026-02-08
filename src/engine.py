from __future__ import annotations
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import compute_metrics
from .robustness import mask_metadata, resize_scale_batch

def class_weights_from_loader(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for batch in loader:
        y = batch["label"]
        for c in range(num_classes):
            counts[c] += (y == c).sum().item()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / counts
    w = w / w.mean()
    return w.to(device)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int,
             meta_level: float = 1.0, res_scale: float = 1.0, base_size: int = 224) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        meta = batch["meta"].to(device)
        y = batch["label"].to(device)

        img = resize_scale_batch(img, float(res_scale), int(base_size))
        meta = mask_metadata(meta, float(meta_level))

        logits = model(img, meta if meta.numel() > 0 else None)
        prob = torch.softmax(logits, dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(prob.argmax(dim=1).cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())

    m = compute_metrics(y_true, y_prob, y_pred, num_classes=num_classes)
    return asdict(m)

def train_one_seed(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    class_balance: bool,
    meta_mask_prob: float,
    resize_augment: bool,
    resize_scales: List[float],
    base_size: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_loader(train_loader, num_classes, device)) if class_balance else nn.CrossEntropyLoss()

    best_val = -1e9
    best_state = None
    bad = 0
    history = {"train_loss": [], "val": []}

    for ep in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train ep {ep}", leave=False):
            img = batch["image"].to(device)
            meta = batch["meta"].to(device)
            y = batch["label"].to(device)

            if resize_augment and len(resize_scales) > 0:
                scale = float(np.random.choice(resize_scales))
                img = resize_scale_batch(img, scale, int(base_size))

            if meta.numel() > 0 and float(meta_mask_prob) > 0:
                if torch.rand(1).item() < float(meta_mask_prob):
                    level = float(np.random.choice([0.75, 0.5, 0.0]))
                    meta = mask_metadata(meta, level)

            opt.zero_grad(set_to_none=True)
            logits = model(img, meta if meta.numel() > 0 else None)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())

        sched.step()
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)

        val_metrics = evaluate(model, val_loader, device, num_classes, meta_level=1.0, res_scale=1.0, base_size=base_size)
        history["val"].append({"epoch": ep, **val_metrics})

        score = val_metrics.get("auroc", np.nan)
        if np.isnan(score):
            score = val_metrics.get("accuracy", 0.0)

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(early_stop_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
