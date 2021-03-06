import torch.nn.functional as F


def xcorr(z, x):
    # cross correlation
    nz = z.size(0)
    nx, c, h, w = x.size()
    x = x.view(-1, nz * c, h, w)
    out = F.conv2d(x, z, groups=nz)
    out = out.view(nx, -1, out.size(-2), out.size(-1))
    return out


def xcorr_depthwise(kernel, x):
    """
        depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
