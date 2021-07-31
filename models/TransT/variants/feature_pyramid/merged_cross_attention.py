import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class PVTMergedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(PVTMergedCrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=to_2tuple(sr_ratio), stride=to_2tuple(sr_ratio))
            self.norm = nn.LayerNorm(dim)

    def forward(self, merged, z_H, z_W, x_H, x_W, x_pos, merged_k_pos):
        B, N, C = merged.shape
        z_size = z_H * z_W
        x = merged[:, z_size:, :]
        q = self.q(x + x_pos).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_size = x_H * x_W
            assert z_size + x_size == N

            z_ = merged[:, : z_size, :]

            z_ = z_.permute(0, 2, 1).reshape(B, C, z_H, z_W)
            z_ = self.sr(z_).reshape(B, C, -1).permute(0, 2, 1)
            z_ = self.norm(z_)

            x_ = x.permute(0, 2, 1).reshape(B, C, x_H, x_W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            merged_downsampled = torch.cat((z_, x_), dim=1)
        else:
            merged_downsampled = merged

        k = self.k(merged_downsampled + merged_k_pos).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(merged_downsampled).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
