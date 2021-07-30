import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class PVTMergedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(PVTMergedSelfAttention, self).__init__()
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

    def forward(self, merged, z_H, z_W, x_H, x_W, merged_q_pos, merged_k_pos):
        B, N, C = merged.shape
        q = self.q(merged + merged_q_pos).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            z_size = z_H * z_W
            x_size = x_H * x_W
            assert z_size + x_size == N

            z = merged[:, : z_size, :]
            x = merged[:, z_size:, :]

            z = z.permute(0, 2, 1).reshape(B, C, z_H, z_W)
            z = self.sr(z).reshape(B, C, -1).permute(0, 2, 1)
            z = self.norm(z)

            x = x.permute(0, 2, 1).reshape(B, C, x_H, x_W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            merged_downsampled = torch.cat((z, x), dim=1)
        else:
            merged_downsampled = merged

        k = self.k(merged_downsampled + merged_k_pos).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(merged_downsampled).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        merged = (attn @ v).transpose(1, 2).reshape(B, N, C)
        merged = self.proj(merged)
        merged = self.proj_drop(merged)

        return merged
