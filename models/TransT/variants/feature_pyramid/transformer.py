import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from typing import Optional


class Transformer(nn.Module):
    def __init__(self, pyramid_modules, pyramid_output_module):
        pass


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
