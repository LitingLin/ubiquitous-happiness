import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.transformer.detr_transformer import TransformerEncoderLayer, TransformerEncoder
from models.transformer.detr_tracking_variants.encoder_shared_params_decoder_cross_attn.decoder import TransformerDecoder


def inference_z_mask_generator(mask_tgt: torch.Tensor):
    return mask_tgt.flatten(1)


def inference_z_x_mask_generator(mask_tgt, mask_src, nhead, dtype):
    mask_src = mask_src.flatten(1)

    n_batch = mask_tgt.shape[0]
    tgt_len = mask_tgt.shape[1]
    src_len = mask_src.shape[1]

    mask_attn = torch.zeros((n_batch, nhead, tgt_len, src_len), dtype=torch.float, device=mask_tgt.device)

    min_float = torch.finfo(dtype).min
    mask_attn.masked_fill_(mask_tgt.unsqueeze(2).unsqueeze(1), min_float)
    mask_attn.masked_fill_(mask_src.unsqueeze(1).unsqueeze(2), min_float)
    return mask_src, mask_attn.view((n_batch * nhead, tgt_len, src_len))


def mask_generator(mask_tgt: torch.Tensor, mask_src: torch.Tensor, nhead: int, dtype):
    """
    :param mask_x (N, H, W)
    :param mask_y (N, H, W)
    """
    assert mask_tgt.shape[0] == mask_src.shape[0]
    assert len(mask_tgt.shape) == 3
    mask_tgt = mask_tgt.flatten(1)
    mask_src = mask_src.flatten(1)

    n_batch = mask_tgt.shape[0]
    tgt_len = mask_tgt.shape[1]
    src_len = mask_src.shape[1]

    mask_attn = torch.zeros((n_batch, nhead, tgt_len, src_len), dtype=torch.float, device=mask_tgt.device)

    min_float = torch.finfo(dtype).min
    mask_attn.masked_fill_(mask_tgt.unsqueeze(2).unsqueeze(1), min_float)
    mask_attn.masked_fill_(mask_src.unsqueeze(1).unsqueeze(2), min_float)
    return mask_tgt, mask_src, mask_attn.view((n_batch * nhead, tgt_len, src_len))


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

        self.d_model = d_model
        self.nhead = nhead

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_template(self, z, z_mask, z_pos):
        z_mask = inference_z_mask_generator(z_mask)
        z, z_pos = flatten_tensor_position_encoding(z, z_pos)
        z = self.encoder(z, src_key_padding_mask=z_mask, pos=z_pos)
        return z, z_mask, z_pos

    def forward_instance(self, z_encoded_flatten, z_mask_flatten, z_pos_flatten, x, x_mask, x_pos):
        x_mask, cross_attn_mask = inference_z_x_mask_generator(z_mask_flatten, x_mask, self.nhead, z_encoded_flatten.dtype)
        x, x_pos = flatten_tensor_position_encoding(x, x_pos)
        x = self.encoder(x, src_key_padding_mask=x_mask, pos=x_pos)
        bbox_embed = self.decoder(z_encoded_flatten, x, z_mask_flatten, cross_attn_mask, z_pos_flatten, x_pos)
        # to N x HW(z) x C
        return bbox_embed.transpose(0, 1)

    def forward(self, z, x, z_mask, x_mask, z_pos, x_pos):
        z_mask, x_mask, cross_attn_mask = mask_generator(z_mask, x_mask, self.nhead, z.dtype)
        z, z_pos = flatten_tensor_position_encoding(z, z_pos)
        x, x_pos = flatten_tensor_position_encoding(x, x_pos)

        z = self.encoder(z, src_key_padding_mask=z_mask, pos=z_pos)
        x = self.encoder(x, src_key_padding_mask=x_mask, pos=x_pos)

        bbox_embed = self.decoder(z, x, z_mask, cross_attn_mask, z_pos, x_pos)
        # to N x HW(z) x C
        return bbox_embed.transpose(0, 1)
