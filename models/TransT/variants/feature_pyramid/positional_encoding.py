import torch.nn.functional as F
from models.TransT.operator.sine_position_encoding import generate_2d_sine_position_encoding, generate_indexed_2d_sine_position_encoding


def generate_transformer_2d_sine_positional_encoding(h, w, dim):
    encoding = generate_2d_sine_position_encoding(1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)


def generate_transformer_indexed_2d_sine_positional_encoding(index, h, w, dim):
    encoding = generate_indexed_2d_sine_position_encoding(index, 1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)


class AbsolutePositionalEncodingGeneratorHelper:
    def __init__(self, base_shape, dim, index, resize_method: str = 'bicubic'):
        if index is not None:
            base_encoding = generate_transformer_indexed_2d_sine_positional_encoding(index, base_shape[1], base_shape[0], dim)
        else:
            base_encoding = generate_transformer_2d_sine_positional_encoding(base_shape[1], base_shape[0], dim)
        base_encoding = base_encoding.view(1, base_shape[1], base_shape[0], dim).permute(0, 3, 1, 2).contiguous()
        self.base_encoding = base_encoding
        self.base_shape = base_shape
        self.dim = dim
        self.resize_method = resize_method

    def generate(self, shape):  # (W, H)
        if shape != self.base_shape:
            encoding = F.interpolate(self.base_encoding, (shape[1], shape[0]), mode=self.resize_method, align_corners=True)
        else:
            encoding = self.base_encoding
        return encoding.permute(0, 2, 3, 1).view(1, shape[0] * shape[1], self.dim).contiguous()
