import torch


def image_numpy_to_torch_HWC_to_CHW(image):
    # HWC -> CHW
    return torch.from_numpy(image).permute((2, 0, 1))


def image_torch_to_numpy_CHW_to_HWC_to_uint8(image: torch.Tensor):
    # CHW -> HWC
    if image.dtype != torch.uint8:
        image = torch.round(image)
        image = image.clamp(min=0, max=255).to(torch.uint8)
    return image.permute((1, 2, 0)).numpy()
