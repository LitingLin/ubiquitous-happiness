import torch


def get_image_mean_nchw(image):
    """
    Args:
        image(torch.Tensor): (n, c, h, w)
    """
    return torch.mean(image, dim=(2, 3))


def get_image_mean_chw(image):
    """
    Args:
        image(torch.Tensor): (c, h, w)
    """
    return torch.mean(image, dim=(1, 2))


def get_image_mean_hw(image):
    """
    Args:
        image(torch.Tensor): (h, w)
    """
    return torch.mean(image)


def get_image_mean(image):
    assert image.ndim in (2, 3, 4)
    if image.ndim == 2:
        return torch.mean(image)
    elif image.ndim == 3:
        return get_image_mean_chw(image)
    else:
        return get_image_mean_nchw(image)
