from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_yaml, save_json, get_device
from src.datasets import SkinDataset, build_encoders
from src.models import RobustMultimodalViT
from src.engine import train_one_seed, evaluate

def build_loaders(cfg_ds: dict, img_size: int, batch_size: int, num_workers: int):
    enc = build_encoders(cfg_ds["csv_train"], cfg_ds["meta_cols"])
    train_ds = SkinDataset(cfg_ds["csv_train"], cfg_ds["root_dir"], cfg_ds["img_col"], cfg_ds["label_col"],
                           cfg_ds["meta_cols"], img_size, encoder=enc, mode="train")
    val_ds = SkinDataset(cfg_ds["csv_val"], cfg_ds["root_dir"], cfg_ds["img_col"], cfg_ds["label_col"],
                         cfg_ds["meta_cols"], img_size, encoder=enc, mode="val")
    test_ds = SkinDataset(cfg_ds["csv_test"], cfg_ds["root_dir"], cfg_ds["img_col"], cfg_ds["label_col"],
                          cfg_ds["meta_cols"], img_size, encoder=enc, mode="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, enc.out_dim()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", required=True, choices=["derm7pt", "pad_ufes_20"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["device"])
    out_dir = os.path.join(cfg["outputs"]["out_dir"], args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    cfg_ds = cfg["data"][args.dataset]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    train_loader, val_loader, test_loader, meta_in_dim = build_loaders(
        cfg_ds, int(model_cfg["img_size"]), int(train_cfg["batch_size"]), int(cfg["num_workers"])
    )

    all_seed_results = []
    for seed in cfg["seed_list"]:
        set_seed(int(seed))
        run_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        model = RobustMultimodalViT(
            img_size=int(model_cfg["img_size"]),
            patch_size=int(model_cfg["patch_size"]),
            num_classes=int(cfg_ds["num_classes"]),
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=float(model_cfg["mlp_ratio"]),
            drop_rate=float(model_cfg["drop_rate"]),
            attn_drop_rate=float(model_cfg["attn_drop_rate"]),
            token_drop_prob=float(model_cfg["token_drop_prob"]),
            meta_in_dim=int(meta_in_dim),
            meta_dim=int(model_cfg["meta_dim"]),
            meta_token=bool(model_cfg["meta_token"]),
            pool=str(model_cfg["pool"]),
        )

        model, history = train_one_seed(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=int(cfg_ds["num_classes"]),
            epochs=int(train_cfg["epochs"]),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            early_stop_patience=int(train_cfg["early_stop_patience"]),
            class_balance=bool(train_cfg["class_balance"]),
            meta_mask_prob=float(train_cfg["meta_mask_prob"]),
            resize_augment=bool(train_cfg["resize_augment"]),
            resize_scales=list(train_cfg["resize_scales"]),
            base_size=int(model_cfg["img_size"]),
        )

        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
        save_json(os.path.join(run_dir, "history.json"), history)

        test_metrics = evaluate(model, test_loader, device, int(cfg_ds["num_classes"]), meta_level=1.0, res_scale=1.0, base_size=int(model_cfg["img_size"]))
        save_json(os.path.join(run_dir, "test_indomain.json"), test_metrics)
        all_seed_results.append(test_metrics)

    keys = list(all_seed_results[0].keys())
    mean_std = {}
    for k in keys:
        vals = [r[k] for r in all_seed_results]
        mean_std[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    save_json(os.path.join(out_dir, "test_indomain_agg.json"), mean_std)
    print("Saved:", os.path.join(out_dir, "test_indomain_agg.json"))

if __name__ == "__main__":
    main()
