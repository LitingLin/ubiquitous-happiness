import torch.nn as nn
from .cross_attention import PVTCrossAttention
from .mlp import Mlp
from timm.models.layers import DropPath


class FeatureFusion(nn.Module):
    def __init__(self, dim_x, dim_y, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(FeatureFusion, self).__init__()
        self.x_q_norm = norm_layer(dim_x)
        self.x_k_norm = norm_layer(dim_y)
        self.y_q_norm = norm_layer(dim_y)
        self.y_k_norm = norm_layer(dim_x)

        self.x_dim_projection = nn.Linear(dim_x, dim_y)
        self.y_dim_projection = nn.Linear(dim_y, dim_x)

        self.x_y_cross_attn = PVTCrossAttention(dim_x, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.y_x_cross_attn = PVTCrossAttention(dim_y, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        x_mlp_hidden_dim = int(dim_x * mlp_ratio)
        y_mlp_hidden_dim = int(dim_y * mlp_ratio)

        self.x_mlp_norm = norm_layer(dim_x)
        self.x_mlp = Mlp(in_features=dim_x, hidden_features=x_mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.y_mlp_norm = norm_layer(dim_y)
        self.y_mlp = Mlp(in_features=dim_y, hidden_features=y_mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, x_H, x_W, y_H, y_W, x_q_pos, x_k_pos, y_q_pos, y_k_pos):
        B, x_N, C = x.shape
        assert x_N == x_H * x_W
        assert B == y.shape[0]

        x_q = x
        x_k = self.x_dim_projection(x)
        y_q = y
        y_k = self.y_dim_projection(y)

        x_q = self.x_q_norm(x_q)
        x_k = self.x_k_norm(x_k)
        y_q = self.y_q_norm(y_q)
        y_k = self.y_k_norm(y_k)

        x = x + self.drop_path(self.x_y_cross_attn(x_q, y_k, y_H, y_W, x_q_pos, y_k_pos))
        y = y + self.drop_path(self.y_x_cross_attn(y_q, x_k, x_H, x_W, y_q_pos, x_k_pos))

        x = x + self.drop_path(self.x_mlp(self.x_mlp_norm(x)))
        y = y + self.drop_path(self.y_mlp(self.y_mlp_norm(y)))

        return x, y
