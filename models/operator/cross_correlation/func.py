import torch.nn.functional as F


def xcorr(z, x):
    # cross correlation
    nz = z.size(0)
    nx, c, h, w = x.size()
    x = x.view(-1, nz * c, h, w)
    out = F.conv2d(x, z, groups=nz)
    out = out.view(nx, -1, out.size(-2), out.size(-1))
    return out


def xcorr_depthwise(z, x):
    r"""
    Depthwise cross correlation. e.g. used for template matching in Siamese tracking network
    Arguments
    ---------
    z: torch.Tensor
        feature_z (N, C, H, W) (e.g. template feature in SOT)
    x: torch.Tensor
        feature_x (N, C, H, W) (e.g. search region feature in SOT)
    Returns
    -------
    torch.Tensor
        cross-correlation result
    """
    batch = int(z.size(0))
    channel = int(z.size(1))
    x = x.view(1, int(batch * channel), int(x.size(2)), int(x.size(3)))
    z = z.view(batch * channel, 1, int(z.size(2)),
               int(z.size(3)))
    out = F.conv2d(x, z, groups=batch * channel)
    out = out.view(batch, channel, int(out.size(2)), int(out.size(3)))
    return out
