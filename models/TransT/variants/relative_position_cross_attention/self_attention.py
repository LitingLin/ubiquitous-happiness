import torch
import torch.nn as nn
from models.backbone.swint.swin_transformer import _generate_2d_relative_position_index, Mlp
from timm.models.layers import DropPath, trunc_normal_


class RelativePositionSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, _2d_shape, # (H, W)
                 qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super(RelativePositionSelfAttention, self).__init__()

        self.register_buffer("relative_position_index", _generate_2d_relative_position_index(_2d_shape))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * _2d_shape[0] - 1) * (2 * _2d_shape[1] - 1), num_heads))

        self.kvq_mat = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B, L, C = qkv.shape

        QKV = self.kvq_mat(qkv).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]

        Q = Q * self.scale
        attn = (Q @ K.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            L, L, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, input_size,  # (H, W)
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SelfAttention, self).__init__()

        self.dim = dim
        self.input_size = input_size

        self.norm1 = norm_layer(dim)

        self.self_attn = RelativePositionSelfAttention(dim, num_heads, input_size, qkv_bias, qk_scale, attn_drop, drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.x_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.x_norm2 = norm_layer(dim)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_shortcut = x

        x = self.norm1(x)

        attn = self.self_attn(x)

        # FFN
        x = x_shortcut + self.x_drop_path(attn)
        x = x + self.x_drop_path(self.x_mlp(self.x_norm2(x)))

        return x
