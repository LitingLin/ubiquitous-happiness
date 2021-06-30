from torchvision.transforms.transforms import Normalize
import torch
from torch import Tensor
from typing import List
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# copy from torchvision.transforms.functional
def denormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    # tensor.sub_(mean).div_(std)
    return tensor


def imagenet_denormalize(tensor: Tensor, inplace: bool = False):
    if not inplace:
        tensor = tensor.clone()
    tensor = denormalize(tensor, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, True)
    tensor *= 255.
    return tensor


def tensor_list_to_cpu(tensor_list):
    return tuple(tensor.cpu() if tensor is not None else None for tensor in tensor_list)
