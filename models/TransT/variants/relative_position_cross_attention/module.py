import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.backbone.swint.swin_transformer import Mlp
from timm.models.layers import DropPath


def generating_2d_relative_position(x_size, y_size):
    x_w, x_h = x_size
    y_w, y_h = y_size
    two_d_positions = []
    for i_x_h in range(x_h):
        for i_x_w in range(x_w):
            positions = []
            for i_y_h in range(y_h):
                y_offset = (i_x_h - i_y_h) * (x_w + y_w - 1)
                for i_y_w in range(y_w):
                    positions.append(i_x_w - i_y_w + y_offset)
            two_d_positions.append(positions)

    two_d_positions = torch.tensor(two_d_positions)
    two_d_positions -= torch.min(two_d_positions)
    return two_d_positions


class RelativePositionCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_2d_shape, x_2d_shape,  # (H, W)
                 qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., relative_position_table_method='native',
                 relative_position_table_init_method='trunc_normal'):
        super(RelativePositionCrossAttention, self).__init__()

        if relative_position_table_method == 'native':
            self.register_buffer("relative_position_index", generating_2d_relative_position((z_2d_shape[1], z_2d_shape[0]),
                                                                                            (x_2d_shape[1], x_2d_shape[0])))
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(((z_2d_shape[0] + x_2d_shape[0] - 1) * (z_2d_shape[1] + x_2d_shape[1] - 1), num_heads)))
        elif relative_position_table_method == 'trunc':
            from models.backbone.swint.swin_transformer import _generate_2d_relative_position_index
            shape = max(z_2d_shape[0], x_2d_shape[0]), max(z_2d_shape[1], x_2d_shape[1])  # (H, W)
            relative_position_index = _generate_2d_relative_position_index(shape)
            relative_position_index_ = []
            for i_z_y in range(z_2d_shape[0]):
                for i_z_x in range(z_2d_shape[1]):
                    relative_position_index__ = []
                    i_z = i_z_x + i_z_y * shape[0]
                    for i_x_y in range(x_2d_shape[0]):
                        i_x_begin = i_x_y * shape[1]
                        i_x_end = i_x_y * shape[1] + x_2d_shape[1]
                        relative_position_index__.append(relative_position_index[i_z, i_x_begin: i_x_end])
                    relative_position_index__ = torch.cat(relative_position_index__)
                    relative_position_index_.append(relative_position_index__)
            relative_position_index = torch.stack(relative_position_index_)
            self.register_buffer("relative_position_index", relative_position_index)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(((shape[0] * 2 - 1) * (shape[1] * 2 - 1), num_heads)))
        else:
            raise NotImplementedError(f"Unknown relative_position_table_method {relative_position_table_method}")

        if relative_position_table_init_method == 'trunc_normal':
            trunc_normal_(self.relative_position_bias_table, std=.02)
        elif relative_position_table_init_method == 'uniform':
            nn.init.uniform_(self.relative_position_bias_table)
        else:
            raise NotImplementedError(f"Unknown relative_position_table_init_method: {relative_position_table_init_method}")

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

        Q = self.q_mat(q).reshape(q_B, q_L, self.num_heads, q_C // self.num_heads).permute(0, 2, 1, 3)
        KV = self.kv_mat(kv).reshape(kv_B, kv_L, 2, self.num_heads, kv_C // self.num_heads).permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1]

        Q = Q * self.scale
        attn = (Q @ K.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            q_L, kv_L, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(q_B, q_L, q_C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size,  # (H, W)
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, relative_position_table_method='native',
                 relative_position_table_init_method='trunc_normal'):
        super(CrossAttention, self).__init__()

        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.z_x_cross_attn = RelativePositionCrossAttention(dim, num_heads, z_size, x_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, relative_position_table_method=relative_position_table_method,
                                                             relative_position_table_init_method=relative_position_table_init_method)
        self.x_z_cross_attn = RelativePositionCrossAttention(dim, num_heads, x_size, z_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, relative_position_table_method=relative_position_table_method,
                                                             relative_position_table_init_method=relative_position_table_init_method)

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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, relative_position_table_method='native',
                 relative_position_table_init_method='trunc_normal'):
        super(CrossAttentionDecoder, self).__init__()

        self.dim = dim
        self.z_size = z_size
        self.x_size = x_size

        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)

        self.x_z_cross_attn = RelativePositionCrossAttention(dim, num_heads, x_size, z_size, qkv_bias, qk_scale,
                                                             attn_drop, drop, relative_position_table_method=relative_position_table_method,
                                                             relative_position_table_init_method=relative_position_table_init_method)

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
