import torch


def bbox_numpy_to_torch(bbox):
    return torch.tensor(bbox)


def bbox_torch_to_numpy(bbox: torch.Tensor):
    return bbox.numpy()
