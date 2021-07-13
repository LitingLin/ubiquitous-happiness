import torch.nn as nn
from timm.models.layers import trunc_normal_
import math


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self.apply(_init_weights)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search, pos_temp, pos_search):
        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.unsqueeze(0)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output
