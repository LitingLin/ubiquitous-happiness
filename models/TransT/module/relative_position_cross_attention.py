import torch
from torch import nn
from models.backbone.swint.swin_transformer import _generate_2d_relative_position_index
from timm.models.layers import trunc_normal_


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
                 proj_drop=0.):
        super(RelativePositionCrossAttention, self).__init__()

        self.register_buffer("relative_position_index", generating_2d_relative_position((z_2d_shape[1], z_2d_shape[0]), (x_2d_shape[1], x_2d_shape[0])))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((z_2d_shape[0] + x_2d_shape[0] - 1) * (z_2d_shape[1] + x_2d_shape[1] - 1), num_heads)))

        self.q_mat = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_mat = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
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
