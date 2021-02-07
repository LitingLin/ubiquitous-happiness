from typing import Optional, List
import torch
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
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, z, x, z_mask, x_mask, z_pos, x_pos):
        z_q = z_k = z + z_pos
        z2 = self.self_attn(z_q, z_k, z, key_padding_mask=z_mask, need_weights=False)[0]
        z = z + self.dropout1(z2)
        z = self.norm1(z)

        z_q = z + z_pos
        x_k = x + x_pos
        z2 = self.multihead_attn(z_q, x_k, x, key_padding_mask=x_mask, need_weights=False)[0]
        z = z + self.dropout2(z2)
        z = self.norm2(z)
        z2 = self.linear2(self.dropout(self.activation(self.linear1(z))))
        z = z + self.dropout3(z2)
        z = self.norm3(z)
        return z


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers):
        super(TransformerDecoder, self).__init__()
        layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers

    def forward(self, z, x, z_mask, x_mask, z_pos, x_pos):
        initial_z = z
        for layer in self.layers:
            z = layer(z, x, z_mask, x_mask, z_pos, x_pos)

        return z - initial_z
