import torch
import torch.nn as nn
from torch.nn import ModuleList
from .cross_attention import AFTCrossAttention, AFTCrossAttentionDecoder
from .self_attention import AFTSelfAttention


class Encoder(nn.Module):
    def __init__(self, dim, z_size, x_size, # (H, W)
                 aft_type):
        super(Encoder, self).__init__()
        self.z_self_attn = AFTSelfAttention(dim, z_size, aft_type)
        self.x_self_attn = AFTSelfAttention(dim, x_size, aft_type)
        self.cross_attn = AFTCrossAttention(dim, z_size, x_size, aft_type)

    def forward(self, z, x):
        z = self.z_self_attn(z)
        x = self.x_self_attn(x)
        z, x = self.cross_attn(z, x)
        return z, x


class Decoder(nn.Module):
    def __init__(self, dim, z_size, x_size, # (H, W)
                 aft_type):
        super(Decoder, self).__init__()
        self.z_self_attn = AFTSelfAttention(dim, z_size, aft_type)
        self.x_self_attn = AFTSelfAttention(dim, x_size, aft_type)
        self.cross_attn = AFTCrossAttentionDecoder(dim, z_size, x_size, aft_type)

    def forward(self, z, x):
        z = self.z_self_attn(z)
        x = self.x_self_attn(x)
        return self.cross_attn(z, x)


class FeatureFusionNetwork(nn.Module):
    def __init__(self, hidden_dim, template_size, search_size, aft_type, n_feature_fusion_layers):
        super(FeatureFusionNetwork, self).__init__()
        self.encoder = ModuleList(
            [Encoder(hidden_dim, template_size, search_size, aft_type) for i in range(n_feature_fusion_layers)]
        )
        self.decoder = Decoder(hidden_dim, template_size, search_size, aft_type)

    def forward(self, z, x):
        z = z.flatten(2).transpose(1, 2)  # N, C, H, W => N, L, C
        x = x.flatten(2).transpose(1, 2)  # N, C, H, W => N, L, C
        for encoder_module in self.encoder:
            z, x = encoder_module(z, x)

        hs = self.decoder(z, x)
        return hs.unsqueeze(0)


def build_feature_fusion_network(network_config: dict):
    transformer_config = network_config['transformer']
    return FeatureFusionNetwork(transformer_config['hidden_dim'],
                                transformer_config['template_size'], transformer_config['search_size'],
                                transformer_config['type'],
                                transformer_config['featurefusion_layers'])
