from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def mask_metadata(meta: torch.Tensor, level: float) -> torch.Tensor:
    if meta.numel() == 0:
        return meta
    if level >= 1.0:
        return meta
    if level <= 0.0:
        return torch.zeros_like(meta)
    B, D = meta.shape
    keep = int(np.ceil(level * D))
    out = meta.clone()
    for i in range(B):
        idx = torch.randperm(D, device=meta.device)
        out[i, idx[keep:]] = 0.0
    return out

def resize_scale_batch(img: torch.Tensor, scale: float, base_size: int) -> torch.Tensor:
    if scale >= 1.0:
        return img
    new_hw = max(1, int(round(base_size * scale)))
    x = F.interpolate(img, size=(new_hw, new_hw), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(base_size, base_size), mode="bilinear", align_corners=False)
    return x
