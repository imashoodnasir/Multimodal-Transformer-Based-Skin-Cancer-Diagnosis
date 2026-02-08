from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_yaml, save_json, get_device
from src.datasets import SkinDataset, build_encoders
from src.models import RobustMultimodalViT
from src.engine import evaluate

def build_loader(cfg_ds: dict, img_size: int, batch_size: int, num_workers: int, split: str):
    enc = build_encoders(cfg_ds["csv_train"], cfg_ds["meta_cols"])
    csv_path = cfg_ds[f"csv_{split}"]
    ds = SkinDataset(csv_path, cfg_ds["root_dir"], cfg_ds["img_col"], cfg_ds["label_col"],
                     cfg_ds["meta_cols"], img_size, encoder=enc, mode="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, enc.out_dim()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_dataset", required=True, choices=["derm7pt", "pad_ufes_20"])
    ap.add_argument("--eval_dataset", required=True, choices=["derm7pt", "pad_ufes_20"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["device"])
    set_seed(args.seed)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    cfg_eval = cfg["data"][args.eval_dataset]

    loader, meta_in_dim = build_loader(cfg_eval, int(model_cfg["img_size"]), int(train_cfg["batch_size"]), int(cfg["num_workers"]), args.split)

    model = RobustMultimodalViT(
        img_size=int(model_cfg["img_size"]),
        patch_size=int(model_cfg["patch_size"]),
        num_classes=int(cfg_eval["num_classes"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        drop_rate=float(model_cfg["drop_rate"]),
        attn_drop_rate=float(model_cfg["attn_drop_rate"]),
        token_drop_prob=0.0,
        meta_in_dim=int(meta_in_dim),
        meta_dim=int(model_cfg["meta_dim"]),
        meta_token=bool(model_cfg["meta_token"]),
        pool=str(model_cfg["pool"]),
    )

    sd = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)

    res_scales = list(cfg["eval"]["resolution_scales"])
    meta_levels = list(cfg["eval"]["meta_levels"])
    base_size = int(model_cfg["img_size"])

    out = {"train_dataset": args.train_dataset, "eval_dataset": args.eval_dataset, "split": args.split, "resolution": {}, "modality": {}, "grid": []}

    for s in res_scales:
        out["resolution"][str(s)] = evaluate(model, loader, device, int(cfg_eval["num_classes"]), meta_level=1.0, res_scale=float(s), base_size=base_size)

    for lv in meta_levels:
        out["modality"][str(lv)] = evaluate(model, loader, device, int(cfg_eval["num_classes"]), meta_level=float(lv), res_scale=1.0, base_size=base_size)

    for s in res_scales:
        for lv in meta_levels:
            m = evaluate(model, loader, device, int(cfg_eval["num_classes"]), meta_level=float(lv), res_scale=float(s), base_size=base_size)
            out["grid"].append({"resolution": float(s), "meta_level": float(lv), **m})

    save_json("robustness_results.json", out)
    print("Saved robustness_results.json")

if __name__ == "__main__":
    main()
