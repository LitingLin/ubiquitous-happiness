import torch
import torch.nn as nn
from torch.nn import ModuleList
from .module import CrossAttention, CrossAttentionDecoder
from .self_attention import SelfAttention


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size, # (H, W)
                 drop_path=0.):
        super(Encoder, self).__init__()
        self.z_self_attn = SelfAttention(dim, num_heads, z_size, drop_path=drop_path)
        self.x_self_attn = SelfAttention(dim, num_heads, x_size, drop_path=drop_path)
        self.cross_attn = CrossAttention(dim, num_heads, z_size, x_size, drop_path=drop_path)

    def forward(self, z, x):
        z = self.z_self_attn(z)
        x = self.x_self_attn(x)
        z, x = self.cross_attn(z, x)
        return z, x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, z_size, x_size, # (H, W)
                 drop_path=0.):
        super(Decoder, self).__init__()
        self.z_self_attn = SelfAttention(dim, num_heads, z_size, drop_path=drop_path)
        self.x_self_attn = SelfAttention(dim, num_heads, x_size, drop_path=drop_path)
        self.cross_attn = CrossAttentionDecoder(dim, num_heads, z_size, x_size, drop_path=drop_path)

    def forward(self, z, x):
        z = self.z_self_attn(z)
        x = self.x_self_attn(x)
        return self.cross_attn(z, x)


class FeatureFusionNetwork(nn.Module):
    def __init__(self, hidden_dim, n_heads, template_size, search_size, n_feature_fusion_layers, drop_path_rate):
        super(FeatureFusionNetwork, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_feature_fusion_layers)]  # stochastic depth decay rule
        self.encoder = ModuleList(
            [Encoder(hidden_dim, n_heads, template_size, search_size, drop_path=dpr[i]) for i in range(n_feature_fusion_layers)]
        )
        self.decoder = Decoder(hidden_dim, n_heads, template_size, search_size)

    def forward(self, z, x):
        z = z.flatten(2).transpose(1, 2)  # N, C, H, W => N, L, C
        x = x.flatten(2).transpose(1, 2)  # N, C, H, W => N, L, C
        for encoder_module in self.encoder:
            z, x = encoder_module(z, x)

        hs = self.decoder(z, x)
        return hs.unsqueeze(0)


def build_feature_fusion_network(network_config: dict):
    transformer_config = network_config['transformer']
    return FeatureFusionNetwork(transformer_config['hidden_dim'], transformer_config['nheads'],
                                transformer_config['template_size'], transformer_config['search_size'],
                                transformer_config['featurefusion_layers'], transformer_config['drop_path_rate'])