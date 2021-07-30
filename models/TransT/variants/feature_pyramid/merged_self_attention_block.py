import torch.nn as nn
from .merged_self_attention import PVTMergedSelfAttention
from .mlp import Mlp
from timm.models.layers import DropPath


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attention_block = PVTMergedSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, merged, z_H, z_W, x_H, x_W, merged_q_pos, merged_k_pos):
        merged = merged + self.drop_path(self.attn(self.norm1(merged), z_H, z_W, x_H, x_W, merged_q_pos, merged_k_pos))
        merged = merged + self.drop_path(self.mlp(self.norm2(merged)))

        return merged
