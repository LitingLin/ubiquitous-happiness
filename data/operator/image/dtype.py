import numpy as np


def image_dtype_float_to_uint8(img: np.ndarray):
    return np.round(img).clip(0., 255.).astype(np.uint8)
