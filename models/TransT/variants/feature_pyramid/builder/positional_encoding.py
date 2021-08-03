import torch.nn as nn


class PositionalEncodingCacheModule(nn.Module):
    def __init__(self, shape_dim_pos_dict):
        super(PositionalEncodingCacheModule, self).__init__()
        for (shape, dim), positional_encoding in shape_dim_pos_dict.items():
            self.register_buffer(f'pos_{shape[0]}x{shape[1]}x{dim}', positional_encoding, False)  # h x w x dim

    def forward(self, shape, dim):
        return getattr(self, f'pos_{shape[0]}x{shape[1]}x{dim}')


def build_positional_encoding(config: dict):
