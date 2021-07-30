import torch
import torch.nn as nn
from .position_encoding import generate_transformer_2d_sine_positional_encoding, \
    generate_transformer_indexed_2d_sine_positional_encoding


def _get_positional_encoding(template_size, search_size, feature_dim, positional_encoding_indexed):
    if positional_encoding_indexed:
        template_positional_encoding = generate_transformer_indexed_2d_sine_positional_encoding(
            0, template_size[1], template_size[0], feature_dim)
        search_positional_encoding = generate_transformer_indexed_2d_sine_positional_encoding(
            1, search_size[1], search_size[0], feature_dim)
    else:
        template_positional_encoding = generate_transformer_2d_sine_positional_encoding(
            template_size[1], template_size[0], feature_dim)
        search_positional_encoding = generate_transformer_2d_sine_positional_encoding(
            search_size[1], search_size[0], feature_dim)
    return template_positional_encoding, search_positional_encoding


class FeatureMerging(nn.Module):
    def __init__(self, template_size, search_size, feature_dim, positional_encoding_indexed):
        super(FeatureMerging, self).__init__()
        self.register_buffer('positional_encoding',
                             torch.cat(_get_positional_encoding(template_size, search_size, feature_dim, positional_encoding_indexed), dim=1), persistent=False)

    def forward(self, z, x):
        return torch.cat((z, x), dim=1), self.positional_encoding


class FeatureUnpacker(nn.Module):
    def __init__(self, template_size, search_size, feature_dim, positional_encoding_indexed):
        super(FeatureUnpacker, self).__init__()
        template_positional_encoding, search_positional_encoding = _get_positional_encoding(template_size, search_size, feature_dim, positional_encoding_indexed)
        self.register_buffer('template_positional_encoding', template_positional_encoding, persistent=False)
        self.register_buffer('search_positional_encoding', search_positional_encoding, persistent=False)
        self.z_size = template_size[0] * template_size[1]
        self.x_size = search_size[0] * search_size[1]

    def forward(self, merged):
        assert merged.shape[1] == self.z_size + self.x_size
        z = merged[:, : self.z_size, :]
        x = merged[:, self.z_size:, :]
        return z, x, self.template_positional_encoding, self.search_positional_encoding


def build_feature_merging(network_config: dict):
    transformer_config = network_config['transformer']
    template_shape = transformer_config['backbone_output_layers']['template']['shape']
    search_shape = transformer_config['backbone_output_layers']['search']['shape']
    dim = transformer_config['hidden_dim']
    position_embedding_indexed = transformer_config['position_embedding']['parameters']['indexed']

    return FeatureMerging(template_shape, search_shape, dim, position_embedding_indexed), FeatureUnpacker(template_shape, search_shape, dim, position_embedding_indexed)
