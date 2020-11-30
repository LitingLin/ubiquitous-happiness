import torch
from torch import nn


def load_weights(net: nn.Module, weight_path: str):
    state_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(state_dict, strict=True)
