import torch
import torchvision.transforms.functional as F


def image_torch_tensor_imagenet_normalize(image: torch.Tensor):
    return F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
