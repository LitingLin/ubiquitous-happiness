import torch
import torch.nn as nn
from .position_encoding import generate_transformer_2d_sine_positional_encoding, \
    generate_transformer_indexed_2d_sine_positional_encoding


class FeatureMerging(nn.Module):
    def __init__(self, template_size_size, search_feature_size, feature_dim, positional_encoding_indexed):
        super(FeatureMerging, self).__init__()
        if positional_encoding_indexed:
            template_positional_encoding = generate_transformer_indexed_2d_sine_positional_encoding(
                0, template_size_size[1], template_size_size[0], feature_dim)
            search_positional_encoding = generate_transformer_indexed_2d_sine_positional_encoding(
                1, search_feature_size[1], search_feature_size[0], feature_dim)
        else:
            template_positional_encoding = generate_transformer_2d_sine_positional_encoding(
                template_size_size[1], template_size_size[0], feature_dim)
            search_positional_encoding = generate_transformer_2d_sine_positional_encoding(
                search_feature_size[1], search_feature_size[0], feature_dim)
        self.register_buffer('positional_encoding',
                             torch.cat((template_positional_encoding, search_positional_encoding), dim=1))

    def forward(self, z, x):
        return torch.cat((z, x), dim=1), self.positional_encoding


def build_feature_merging(network_config: dict):
    transformer_config = network_config['transformer']
    template_shape = transformer_config['backbone_output_layers']['template']['shape']
    search_shape = transformer_config['backbone_output_layers']['search']['shape']
    dim = transformer_config['hidden_dim']
    position_embedding_indexed = transformer_config['position_embedding']['parameters']['indexed']

    return FeatureMerging(template_shape, search_shape, dim, position_embedding_indexed)
