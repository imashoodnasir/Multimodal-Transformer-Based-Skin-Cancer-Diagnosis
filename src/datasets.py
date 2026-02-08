from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _safe_join(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)

class TabularEncoder:
    def __init__(self, meta_cols: List[str]):
        self.meta_cols = meta_cols
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.cat_maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for c in self.meta_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                self.num_cols.append(c)
            else:
                self.cat_cols.append(c)

        for c in self.num_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x) + 1e-6)
            self.means[c] = mu
            self.stds[c] = sd

        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("NA").values.tolist()
            uniq = sorted(set(vals))
            m = {v: i + 1 for i, v in enumerate(uniq)}
            self.cat_maps[c] = m

    def transform_row(self, row: pd.Series) -> np.ndarray:
        feats = []
        for c in self.num_cols:
            v = row.get(c, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if np.isnan(v):
                v = self.means[c]
            v = (v - self.means[c]) / self.stds[c]
            feats.append(v)

        for c in self.cat_cols:
            v = row.get(c, "NA")
            v = "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            idx = self.cat_maps[c].get(v, 0)
            feats.append(float(idx))

        return np.asarray(feats, dtype=np.float32)

    def out_dim(self) -> int:
        return len(self.num_cols) + len(self.cat_cols)

class SkinDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        img_col: str,
        label_col: str,
        meta_cols: List[str],
        img_size: int,
        encoder: Optional[TabularEncoder] = None,
        mode: str = "train",
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        self.meta_cols = meta_cols
        self.encoder = encoder
        self.mode = mode

        if mode == "train":
            self.base_tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.base_tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = _safe_join(self.root_dir, str(row[self.img_col]))
        img = Image.open(img_path).convert("RGB")
        img = self.base_tf(img)

        y = int(row[self.label_col])

        if self.encoder is None or len(self.meta_cols) == 0:
            meta = np.zeros((0,), dtype=np.float32)
        else:
            meta = self.encoder.transform_row(row)

        return {
            "image": img,
            "meta": torch.tensor(meta, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }

def build_encoders(train_csv: str, meta_cols: List[str]) -> TabularEncoder:
    df = pd.read_csv(train_csv)
    enc = TabularEncoder(meta_cols=meta_cols)
    if len(meta_cols) > 0:
        for c in meta_cols:
            if c not in df.columns:
                raise ValueError(f"Missing metadata column '{c}' in {train_csv}")
        enc.fit(df)
    return enc
