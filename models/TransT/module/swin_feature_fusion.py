import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from .relative_position_cross_attention import RelativePositionCrossAttention
from models.backbone.swint.swin_transformer import Mlp
import math


def patch_partition(x, patch_size):
    """
    Args:
        x: (B, H, W, C)
        patch_size (int): patch size
    Returns:
        patches: (B, H_nW, W_nW, patch_size, patch_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return patches


def patch_reverse(patches):
    """
    Args:
        patches: (num_windows*B, window_size, window_size, C) (B, H_nW, W_nW, patch_size, patch_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B, H_nW, W_nW, patch_size, patch_size, C = patches.shape
    x = patches.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_nW * patch_size, W_nW * patch_size, C)
    return x


def _prepare_patch_partition(x, H, W, patch_size):
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    x = x.view(B, H, W, C)

    # pad feature maps to multiples of patch size
    pad_l = pad_t = 0
    pad_r = (patch_size - W % patch_size) % patch_size
    pad_b = (patch_size - H % patch_size) % patch_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

    # partition patches
    x_patches = patch_partition(x, patch_size)  # (B, H_nW, W_nW, patch_size, patch_size, C)
    _, H_nP, W_nP, _, _, _ = x_patches.shape
    x_patches = x_patches.view(B, H_nP * W_nP, patch_size * patch_size * C)  # (B, H_nP * W_nP, patch_size * patch_size * C)

    return x_patches


def _reverse_patch_partition(patches, H, W, H_num_patches, W_num_patches, patch_size, C):
    # merge patches
    B, _, _, = patches.shape
    patches = patches.view(B, H_num_patches, W_num_patches, patch_size, patch_size, C)
    x = patch_reverse(patches)  # B H' W' C

    _, Hp, Wp, _ = x.shape

    if Hp != H or Wp != W:
        x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)
    return x


class InterPatchCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size,  # (H, W)
                 patch_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(InterPatchCrossAttention, self).__init__()
        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size
        self.patch_size = patch_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.z_num_patches = int(math.ceil(z_size[0] / patch_size)), int(math.ceil(z_size[1] / patch_size))  # (H, W)
        self.x_num_patches = int(math.ceil(x_size[0] / patch_size)), int(math.ceil(x_size[1] / patch_size))  # (H, W)

        cross_patch_attn_dim = dim * patch_size * patch_size

        self.z_x_cross_attn = RelativePositionCrossAttention(cross_patch_attn_dim, num_heads, self.z_num_patches, self.x_num_patches, qkv_bias, qk_scale, attn_drop, drop)
        self.x_z_cross_attn = RelativePositionCrossAttention(cross_patch_attn_dim, num_heads, self.x_num_patches, self.z_num_patches, qkv_bias,
                                                             qk_scale, attn_drop, drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.z_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.z_norm2 = norm_layer(dim)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.x_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.x_norm2 = norm_layer(dim)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        z_H, z_W = self.z_size
        x_H, x_W = self.x_size

        z_shortcut = z
        x_shortcut = x

        z = self.z_norm1(z)
        x = self.x_norm1(x)

        z_windows = _prepare_patch_partition(z, z_H, z_W, self.patch_size)
        x_windows = _prepare_patch_partition(x, x_H, x_W, self.patch_size)

        z_x_attn = self.z_x_cross_attn(z_windows, x_windows)
        x_z_attn = self.x_z_cross_attn(x_windows, z_windows)

        z_x_attn = _reverse_patch_partition(z_x_attn, z_H, z_W, self.z_num_patches[0], self.z_num_patches[1],
                                            self.patch_size, self.dim)
        x_z_attn = _reverse_patch_partition(x_z_attn, x_H, x_W, self.x_num_patches[0], self.x_num_patches[1],
                                            self.patch_size, self.dim)

        # FFN
        z = z_shortcut + self.z_drop_path(z_x_attn)
        z = z + self.z_drop_path(self.z_mlp(self.z_norm2(z)))

        x = x_shortcut + self.x_drop_path(x_z_attn)
        x = x + self.x_drop_path(self.x_mlp(self.x_norm2(x)))

        return z, x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size,  # (H, W)
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossAttention, self).__init__()

        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.z_x_cross_attn = RelativePositionCrossAttention(dim, num_heads, z_size, x_size, qkv_bias, qk_scale, attn_drop, drop)
        self.x_z_cross_attn = RelativePositionCrossAttention(dim, num_heads, x_size, z_size, qkv_bias, qk_scale, attn_drop, drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.z_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.z_norm2 = norm_layer(dim)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.x_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.x_norm2 = norm_layer(dim)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        z_shortcut = z
        x_shortcut = x

        z = self.z_norm1(z)
        x = self.x_norm1(x)

        z_x_attn = self.z_x_cross_attn(z, x)
        x_z_attn = self.x_z_cross_attn(x, z)

        # FFN
        z = z_shortcut + self.z_drop_path(z_x_attn)
        z = z + self.z_drop_path(self.z_mlp(self.z_norm2(z)))

        x = x_shortcut + self.x_drop_path(x_z_attn)
        x = x + self.x_drop_path(self.x_mlp(self.x_norm2(x)))

        return z, x
