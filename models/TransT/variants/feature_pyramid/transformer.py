import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from typing import Optional



'''Pyramid module:
        level_0_self_attention_q_positional_encodings:
            level 0 z+x
        level_0_self_attention_k_positional_encodings:
            level 0 z+x / sr_ratio
        level_1_self_attention_q_positional_encodings:
            level 1 z+x
        level_1_self_attention_k_positional_encodings:
            level 1 z+x / sr_ratio
        level_0_z_feature_fusion_q_positional_encoding, level_0_z_feature_fusion_k_positional_encoding,
        level_1_z_feature_fusion_q_positional_encoding, level_1_z_feature_fusion_k_positional_encoding,
        level_0_x_feature_fusion_q_positional_encoding, level_0_x_feature_fusion_k_positional_encoding,
        level_1_x_feature_fusion_q_positional_encoding, level_1_x_feature_fusion_k_positional_encoding        
'''

class Transformer(nn.Module):
    def __init__(self, backbone, z_backbone_output_stages, x_backbone_output_stages, z_feature_map_sizes, x_feature_map_sizes,
                 positional_encoding_provider, pyramid_modules, level_0_decoder, level_1_decoder, head):
        super(Transformer, self).__init__()
        self.backbone = backbone
        self.z_backbone_output_stages = z_backbone_output_stages
        self.x_backbone_output_stages = x_backbone_output_stages
        self.positional_encoding_provider = positional_encoding_provider

        self.z_positional_encodings = z_positional_encodings
        self.x_positional_encodings = x_positional_encodings
        self.merged_positional_encodings =

        self.z_feature_map_sizes = z_feature_map_sizes
        self.x_feature_map_sizes = x_feature_map_sizes

        self.pyramid_modules = nn.ModuleList(pyramid_modules)
        self.pyramid_modules_positional_encoding_shapes = {}

        self.level_0_decoder = level_0_decoder
        self.level_1_decoder = level_1_decoder

        self.head = head


    def forward(self, z, x):
        z_level_0, z_level_1 = self.backbone(z, self.z_backbone_output_stages, False)
        x_level_0, x_level_1 = self.backbone(x, self.x_backbone_output_stages, False)

        for pyramid_module in self.pyramid_modules:
            pyramid_module(z_level_0, x_level_0, self.z_feature_map_sizes[0][1], self.z_feature_map_sizes[0][0], self.x_feature_map_sizes[0][1], self.x_feature_map_sizes[0][0],
                           z_level_1, x_level_1, self.z_feature_map_sizes[1][1], self.z_feature_map_sizes[1][0], self.x_feature_map_sizes[1][1], self.x_feature_map_sizes[1][0],
                           )




def build_transformer(network_config: dict):
    transformer_config = network_config['transformer']
    return Transformer(
        d_model=transformer_config['hidden_dim'],
        dropout=transformer_config['dropout'],
        nhead=transformer_config['num_heads'],
        dim_feedforward=transformer_config['dim_feedforward'],
        num_encoder_layers=transformer_config['encoder_num_layers'],
        num_decoder_layers=transformer_config['decoder_num_layers'],
        activation=transformer_config['activation']
    )
