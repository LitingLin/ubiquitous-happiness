import torch.nn as nn
from .cross_attention import PVTCrossAttention
from .self_attention import PVTSelfAttention
from .mlp import Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class FeatureFusionEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(FeatureFusionEncoderLayer, self).__init__()
        self.x_self_norm = norm_layer(dim)
        self.x_self_attn = PVTSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        self.y_self_norm = norm_layer(dim)
        self.y_self_attn = PVTSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        self.x_cross_norm = norm_layer(dim)
        self.y_cross_norm = norm_layer(dim)
        self.x_y_cross_attn = PVTCrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.y_x_cross_attn = PVTCrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.x_mlp_norm = norm_layer(dim)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.y_mlp_norm = norm_layer(dim)
        self.y_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, x_H, x_W, y_H, y_W):
        x = x + self.drop_path(self.x_self_attn(self.x_self_norm(x), x_H, x_W))
        y = y + self.drop_path(self.y_self_attn(self.y_self_norm(x), y_H, y_W))

        x_ = self.x_cross_norm(x)
        y_ = self.y_cross_norm(y)
        x = x + self.drop_path(self.x_y_cross_attn(x_, y_, y_H, y_W))
        y = y + self.drop_path(self.y_x_cross_attn(y_, x_, x_H, x_W))

        x = x + self.drop_path(self.x_mlp(self.x_mlp_norm(x), x_H, x_W))
        y = y + self.drop_path(self.y_mlp(self.y_mlp_norm(y), y_H, y_W))
        return x, y


class FeatureFusionDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(FeatureFusionDecoderLayer, self).__init__()
        self.x_self_norm = norm_layer(dim)
        self.x_self_attn = PVTSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        self.y_self_norm = norm_layer(dim)
        self.y_self_attn = PVTSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        self.x_cross_norm = norm_layer(dim)
        self.y_cross_norm = norm_layer(dim)
        self.y_x_cross_attn = PVTCrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.y_mlp_norm = norm_layer(dim)
        self.y_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, x_H, x_W, y_H, y_W):
        x = x + self.drop_path(self.x_self_attn(self.x_self_norm(x), x_H, x_W))
        y = y + self.drop_path(self.y_self_attn(self.y_self_norm(x), y_H, y_W))

        x_ = self.x_cross_norm(x)
        y_ = self.y_cross_norm(y)
        y = y + self.drop_path(self.y_x_cross_attn(y_, x_, x_H, x_W))

        y = y + self.drop_path(self.y_mlp(self.y_mlp_norm(y), y_H, y_W))
        return y
