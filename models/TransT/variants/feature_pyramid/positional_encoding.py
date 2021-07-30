import torch.nn as nn
import torch.nn.functional as F
from models.TransT.operator.sine_position_encoding import generate_2d_sine_position_encoding, generate_indexed_2d_sine_position_encoding


def generate_transformer_2d_sine_positional_encoding(h, w, dim):
    encoding = generate_2d_sine_position_encoding(1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)


def generate_transformer_indexed_2d_sine_positional_encoding(index, h, w, dim):
    encoding = generate_indexed_2d_sine_position_encoding(index, 1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)


class PositionalEncoding(nn.Module):
    def __init__(self, base_shape, dim, resized_shapes = None, resize_method: str = 'bicubic'):
        super(PositionalEncoding, self).__init__()
        base_positional_encoding = generate_transformer_2d_sine_positional_encoding(base_shape[1], base_shape[0], dim)

        self.register_buffer(f'pos_{base_shape[1]}x{base_shape[0]}', base_positional_encoding)

        if resized_shapes is not None and len(resized_shapes) > 0:
            base_positional_encoding_reshaped = base_positional_encoding.view(1, base_shape[1], base_shape[0], dim)
            base_positional_encoding_reshaped = base_positional_encoding_reshaped.permute(0, 3, 1, 2)

            for resized_shape in resized_shapes:
                resized_positional_encoding = F.interpolate(base_positional_encoding_reshaped, (resized_shape[1], resized_shape[0]), mode=resize_method, align_corners=True)
                resized_positional_encoding = resized_positional_encoding.permute(0, 2, 3, 1).view(1, resized_shape[0] * resized_shape[1], dim)
                self.register_buffer(f'pos_{resized_shape[1]}x{resized_shape[0]}', resized_positional_encoding)

    def forward(self, shape):
        name = f'pos_{shape[0]}x{shape[1]}'
        positional_encoding = getattr(self, name)
        return positional_encoding
