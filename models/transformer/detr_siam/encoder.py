from typing import Optional, List
from torch import nn, Tensor
import torch.nn.functional as F
import copy


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,
                z, x,
                z_key_padding_mask: Optional[Tensor] = None,
                x_key_padding_mask: Optional[Tensor] = None,
                z_pos: Optional[Tensor] = None,
                x_pos: Optional[Tensor] = None):
        for layer in self.layers:
            z, x = layer(z, x,
                z_key_padding_mask,
                x_key_padding_mask,
                z_pos,
                x_pos)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn_z = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_x = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.z_norm = nn.LayerNorm(d_model)
        self.z_dropout = nn.Dropout(dropout)

        self.x_norm = nn.LayerNorm(d_model)
        self.x_dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model)

        self.z_ffn_linear1 = nn.Linear(d_model, dim_feedforward)
        self.z_ffn_dropout_1 = nn.Linear(d_model, dim_feedforward)
        self.z_ffn_linear2 = nn.Linear(d_model, dim_feedforward)
        self.z_ffn_dropout_2 = nn.Linear(d_model, dim_feedforward)
        self.z_ffn_activation = _get_activation_fn(activation)
        self.z_ffn_norm = nn.LayerNorm(d_model)

        self.x_ffn_linear1 = nn.Linear(d_model, dim_feedforward)
        self.x_ffn_dropout_1 = nn.Linear(d_model, dim_feedforward)
        self.x_ffn_linear2 = nn.Linear(d_model, dim_feedforward)
        self.x_ffn_dropout_2 = nn.Linear(d_model, dim_feedforward)
        self.x_ffn_activation = _get_activation_fn(activation)
        self.x_ffn_norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def x_forward(self, x, x_key_padding_mask, x_pos):
        x_q = x_k = self.with_pos_embed(x, x_pos)
        x_2 = self.self_attn_x(x_q, x_k, value=x, key_padding_mask=x_key_padding_mask)[0]
        x = x + self.x_dropout(x_2)
        x = self.x_norm(x)
        return x

    def z_forward(self, z, z_key_padding_mask, z_pos):
        z_q = z_k = self.with_pos_embed(z, z_pos)
        z_2 = self.self_attn_z(z_q, z_k, value=z, key_padding_mask=z_key_padding_mask)[0]
        z = z + self.z_dropout(z_2)
        z = self.z_norm(z)
        return z

    # def z_x_cross_attn(self, z, z_pos, x, x_pos, x_key_padding_mask):
    #     x_2 = self.cross_attn(self.with_pos_embed(z, z_pos), self.with_pos_embed(x, x_pos), value=x, key_padding_mask=x_key_padding_mask)[0]
    #     x = x + self.cross_dropout(x_2)
    #     x = self.cross_norm(x)
    #     return x

    def z_x_cross_attn(self, z, z_pos, x, x_pos, z_key_padding_mask):
        z_2 = self.cross_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(z, z_pos), value=z, key_padding_mask=z_key_padding_mask)[0]
        x = x + self.cross_dropout(z_2)
        x = self.cross_norm(x)
        return x

    def z_ffn(self, z):
        z_2 = self.z_ffn_linear2(self.z_ffn_dropout_1(self.z_ffn_activation(self.z_ffn_linear1(z))))
        z = z + self.z_ffn_dropout_2(z_2)
        z = self.z_ffn_norm(z)
        return z

    def x_ffn(self, x):
        x_2 = self.x_ffn_linear2(self.x_ffn_dropout_1(self.x_ffn_activation(self.x_ffn_linear1(x))))
        x = x + self.x_ffn_dropout_2(x_2)
        x = self.x_ffn_norm(x)
        return x

    def forward(self,
                z, x,
                z_key_padding_mask: Optional[Tensor] = None,
                x_key_padding_mask: Optional[Tensor] = None,
                z_pos: Optional[Tensor] = None,
                x_pos: Optional[Tensor] = None):
        z = self.z_forward(z, z_key_padding_mask, z_pos)
        x = self.x_forward(x,  x_key_padding_mask, x_pos)

        x = self.z_x_cross_attn(z, z_pos, x, x_pos, z_key_padding_mask)

        x = self.cross_attn(self.with_pos_embed(z, z_pos), self.with_pos_embed(x, x_pos), value=x, key_padding_mask=x_key_padding_mask)[0]

        z = self.z_ffn(z)
        x = self.x_ffn(x)

        return z, x
