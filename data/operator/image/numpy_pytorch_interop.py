import torch


def image_numpy_to_torch(image):
    # HWC -> CHW
    return torch.from_numpy(image).permute((2, 0, 1)).float()


def image_torch_to_numpy(image: torch.Tensor):
    # CHW -> HWC
    return image.permute((1, 2, 0)).clamp(min=0, max=255).to(dtype=torch.uint8).numpy()
