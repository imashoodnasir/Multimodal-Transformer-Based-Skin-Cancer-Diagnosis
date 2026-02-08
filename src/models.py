from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, meta_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, embed_dim),
        )

    def forward(self, m):
        return self.net(m)

class RobustMultimodalViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        token_drop_prob: float,
        meta_in_dim: int,
        meta_dim: int,
        meta_token: bool,
        pool: str,
    ):
        super().__init__()
        self.pool = pool
        self.token_drop_prob = float(token_drop_prob)

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_meta = bool(meta_token) and int(meta_in_dim) > 0
        self.meta_proj = MetaMLP(int(meta_in_dim), int(meta_dim), embed_dim) if self.use_meta else None

        seq_len = 1 + num_patches + (1 if self.use_meta else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=float(drop_rate))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, float(mlp_ratio), float(drop_rate), float(attn_drop_rate))
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _token_drop(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.token_drop_prob <= 0:
            return x
        B, N, D = x.shape
        has_meta = self.use_meta
        cls = x[:, :1, :]
        if has_meta:
            meta = x[:, -1:, :]
            patches = x[:, 1:-1, :]
        else:
            meta = None
            patches = x[:, 1:, :]

        keep_prob = 1.0 - self.token_drop_prob
        mask = (torch.rand(B, patches.shape[1], device=x.device) < keep_prob).float()
        token_count = mask.sum(dim=1, keepdim=True)
        mask = torch.where(token_count > 0, mask, torch.ones_like(mask))
        patches = patches * mask.unsqueeze(-1)

        if has_meta:
            return torch.cat([cls, patches, meta], dim=1)
        return torch.cat([cls, patches], dim=1)

    def forward(self, img: torch.Tensor, meta: Optional[torch.Tensor] = None):
        B = img.shape[0]
        x = self.patch_embed(img)
        cls = self.cls_token.expand(B, -1, -1)

        tokens = [cls, x]
        if self.use_meta:
            if meta is None:
                raise ValueError("Metadata token enabled but meta is None.")
            mt = self.meta_proj(meta).unsqueeze(1)
            tokens.append(mt)

        x = torch.cat(tokens, dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)
        x = self._token_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        feat = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.head(feat)
