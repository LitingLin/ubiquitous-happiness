import torch


def to_torch_tensor(image):
    return torch.from_numpy(image).permute((2, 0, 1))
