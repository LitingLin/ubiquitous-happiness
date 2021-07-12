import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.backbone.swint.swin_transformer import Mlp
from timm.models.layers import DropPath


class RelativePositionCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_2d_shape, x_2d_shape,  # (H, W)
                 qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., position_encoder_spatial_locality=3, position_encoder_dim_locality=4):
        super(RelativePositionCrossAttention, self).__init__()

        assert dim % position_encoder_dim_locality == 0
        self.q_position_encoder = nn.Conv2d(dim, dim, position_encoder_spatial_locality, padding=position_encoder_spatial_locality - 2, groups=dim // position_encoder_dim_locality)
        self.k_position_encoder = nn.Conv2d(dim, dim, position_encoder_spatial_locality, padding=position_encoder_spatial_locality - 2, groups=dim // position_encoder_dim_locality)

        self.z_2d_shape = z_2d_shape
        self.x_2d_shape = x_2d_shape

        self.q_mat = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_mat = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv):
        q_B, q_L, q_C = q.shape
        kv_B, kv_L, kv_C = kv.shape

        Q = self.q_mat(q)
        KV = self.kv_mat(kv)

        KV = KV.view(kv_B, kv_L, 2, kv_C)
        K, V = KV.unbind(2)

        q_enc = self.q_position_encoder(
            Q.transpose(1, 2).view(q_B, q_C, *self.z_2d_shape)).view(q_B, q_C, -1).transpose(1, 2)
        k_enc = self.k_position_encoder(
            K.transpose(1, 2).view(kv_B, kv_C, *self.x_2d_shape)).view(kv_B, kv_C, -1).transpose(1, 2)

        Q = (Q + q_enc).reshape(q_B, q_L, self.num_heads, q_C // self.num_heads).permute(0, 2, 1, 3)
        K = (K + k_enc).reshape(kv_B, kv_L, self.num_heads, kv_C // self.num_heads).permute(0, 2, 1, 3)
        V = V.reshape(kv_B, kv_L, self.num_heads, kv_C // self.num_heads).permute(0, 2, 1, 3)

        Q = Q * self.scale
        attn = (Q @ K.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(q_B, q_L, q_C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size,  # (H, W)
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, position_encoder_spatial_locality=3, position_encoder_dim_locality=4):
        super(CrossAttention, self).__init__()

        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.z_x_cross_attn = RelativePositionCrossAttention(dim, num_heads, z_size, x_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, position_encoder_spatial_locality, position_encoder_dim_locality)
        self.x_z_cross_attn = RelativePositionCrossAttention(dim, num_heads, x_size, z_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, position_encoder_spatial_locality, position_encoder_dim_locality)

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


class CrossAttentionDecoder(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size,  # (H, W)
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 position_encoder_spatial_locality=3, position_encoder_dim_locality=4):
        super(CrossAttentionDecoder, self).__init__()

        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.x_z_cross_attn = RelativePositionCrossAttention(dim, num_heads, x_size, z_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, position_encoder_spatial_locality, position_encoder_dim_locality)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.x_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.x_norm2 = norm_layer(dim)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        x_shortcut = x

        z = self.z_norm1(z)
        x = self.x_norm1(x)

        x_z_attn = self.x_z_cross_attn(x, z)

        # FFN
        x = x_shortcut + self.x_drop_path(x_z_attn)
        x = x + self.x_drop_path(self.x_mlp(self.x_norm2(x)))

        return x
