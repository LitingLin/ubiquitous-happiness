import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.transformer.detr_transformer import TransformerEncoderLayer, TransformerEncoder
from models.transformer.detr_tracking_variants.encoder_shared_params_decoder_cross_attn.decoder import TransformerDecoder


def mask_generator(mask_x: torch.BoolTensor, mask_y: torch.BoolTensor):
    """
    :param mask_x (N, H, W)
    :param mask_y (N, H, W)
    """
    assert mask_x.shape == mask_y.shape
    assert len(mask_x.shape) == 3
    mask_x = mask_x.flatten(1)
    mask_y = mask_y.flatten(1)
    mask_attn = ~torch.matmul((~mask_x.unsqueeze(1).transpose(1, 2)).to(torch.int), (~mask_y.unsqueeze(1)).to(torch.int)).to(torch.bool)

    return mask_x, mask_y, mask_attn


def flatten_tensor_position_encoding(tensor, position):
    # flatten NxCxHxW to HWxNxC
    tensor = tensor.flatten(2).permute(2, 0, 1)
    position = position.flatten(2).permute(2, 0, 1)
    return tensor, position


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, False)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)

        self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward, dropout, activation, num_decoder_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x, z_mask, x_mask, z_pos, x_pos):
        z_mask, x_mask, cross_attn_mask = mask_generator(z_mask, x_mask)
        z, z_pos = flatten_tensor_position_encoding(z, z_pos)
        x, x_pos = flatten_tensor_position_encoding(x, x_pos)

        z = self.encoder(z, src_key_padding_mask=z_mask, pos=z_pos)
        x = self.encoder(x, src_key_padding_mask=x_mask, pos=x_pos)

        bbox_embed = self.decoder(z, x, z_mask, cross_attn_mask, z_pos, x_pos)
        # to N x HW(z) x C
        return bbox_embed.transpose(0, 1)
