from models.TransT.operator.sine_position_encoding import generate_2d_sine_position_encoding, generate_indexed_2d_sine_position_encoding


def generate_transformer_2d_sine_positional_encoding(h, w, dim):
    encoding = generate_2d_sine_position_encoding(1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)


def generate_transformer_indexed_2d_sine_positional_encoding(index, h, w, dim):
    encoding = generate_indexed_2d_sine_position_encoding(index, 1, h, w, dim)
    return encoding.view(1, h * w, dim)  # (N, L, C)
