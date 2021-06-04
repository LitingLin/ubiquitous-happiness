import torch
from data.operator.


def scaling_to_multi_resolution(image: torch.Tensor, resolutions):
    c, h, w = image.shape

    size = w * h
    scaling = tuple(resolution / size for resolution in resolutions)
    level = scaling.index(min(scaling))


    pass