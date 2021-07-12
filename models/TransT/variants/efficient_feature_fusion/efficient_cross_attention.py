import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _forward_self_attn_efficient(x, pos_x, efficient_pos_x, efficient_object, attention, dropout, norm):
    q = x + pos_x
    kv = efficient_object(x)
    k = kv + efficient_pos_x
    v = kv
    attn = attention(q, k, value=v)[0]
    x = x + dropout(attn)
    x = norm(x)
    return x


def _forward_cross_attn_efficient(q, kv, q_pos, k_efficient_pos, efficient_object, attention, dropout, norm):
    q_encoded = q + q_pos
    kv = efficient_object(kv)
    k_encoded = kv + k_efficient_pos
    v = kv
    attn = attention(query=q_encoded, key=k_encoded, value=v)[0]
    q = q + dropout(attn)
    q = norm(q)
    return q


def _feed_forward_network(x, linear_1, activation):
    src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
    src1 = src1 + self.dropout13(src12)
    src1 = self.norm13(src1)



class EfficientCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", k=2):
        super(EfficientCrossAttention, self).__init__()

        self.self_attn1_efficient = nn.Conv2d(d_model, d_model, k, k)
        self.self_attn2_efficient = nn.Conv2d(d_model, d_model, k, k)
        self.cross_attn1_efficient = nn.Conv2d(d_model, d_model, k, k)
        self.cross_attn2_efficient = nn.Conv2d(d_model, d_model, k, k)

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def forward(self, src1, src2,
                pos_src1,
                pos_src2,
                efficient_pos_src1, efficient_pos_src2):
        src1 = _forward_self_attn_efficient(src1, pos_src1, efficient_pos_src1, self.self_attn1_efficient,
                                            self.self_attn1, self.dropout11, self.norm11)

        src2 = _forward_self_attn_efficient(src2, pos_src2, efficient_pos_src2, self.self_attn2_efficient,
                                            self.self_attn2, self.dropout21, self.norm21)

        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                     key=self.with_pos_embed(src2, pos_src2),
                                     value=src2)[0]
        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                     key=self.with_pos_embed(src1, pos_src1),
                                     value=src1)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2
