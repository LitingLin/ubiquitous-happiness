import torch
from torch import nn, Tensor
from .encoder import TransformerEncoder, TransformerEncoderLayer
from models.transformer.detr_transformer import TransformerDecoder, TransformerDecoderLayer
from typing import Optional


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=False):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_embed,
                z, x,
                z_mask: Optional[Tensor] = None,
                x_mask: Optional[Tensor] = None,
                z_pos: Optional[Tensor] = None,
                x_pos: Optional[Tensor] = None, ):

        # flatten NxCxHxW to HWxNxC
        z_bs, z_c, z_h, z_w = z.shape
        x_bs, x_c, x_h, x_w = x.shape
        assert z_bs == x_bs
        assert z_c == x_c

        z = z.flatten(2).permute(2, 0, 1)
        x = x.flatten(2).permute(2, 0, 1)

        z_pos = z_pos.flatten(2).permute(2, 0, 1)
        x_pos = x_pos.flatten(2).permute(2, 0, 1)

        z_mask = z_mask.flatten(1)
        x_mask = x_mask.flatten(1)

        query_embed = query_embed.unsqueeze(1).repeat(1, x_bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(z, x, z_mask, x_mask, z_pos, x_pos)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=x_mask,
                          pos=x_pos, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(x_bs, x_c, x_h, x_w)


def build_transformer(net_config: dict):
    transformer_config = net_config['transformer']
    return Transformer(
        d_model=transformer_config['hidden_dim'],
        dropout=transformer_config['dropout'],
        nhead=transformer_config['num_heads'],
        dim_feedforward=transformer_config['feed_forward']['dim'],
        num_encoder_layers=transformer_config['encoder']['num_layers'],
        num_decoder_layers=transformer_config['decoder']['num_layers'],
        return_intermediate_dec=True,
    )
