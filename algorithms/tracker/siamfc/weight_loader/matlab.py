import scipy.io
import torch
from torch import nn
import numpy as np
from models.head.siamfc import SiamFCLinearHead, SiamFCBNHead


# https://github.com/albanie/pytorch-mcn/blob/master/python/ptmcn_utils.py
def weights2tensor(x, squeeze=False, in_features=None, out_features=None):
    """Adjust memory layout and load weights as torch tensor
    Args:
        x (ndaray): a numpy array, corresponding to a set of network weights
           stored in column major order
        squeeze (bool) [False]: whether to squeeze the tensor (i.e. remove
           singletons from the trailing dimensions. So after converting to
           pytorch layout (C_out, C_in, H, W), if the shape is (A, B, 1, 1)
           it will be reshaped to a matrix with shape (A,B).
        in_features (int :: None): used to reshape weights for a linear block.
        out_features (int :: None): used to reshape weights for a linear block.
    Returns:
        torch.tensor: a permuted sets of weights, matching the pytorch layout
        convention
    """
    if x.ndim == 4:
        x = x.transpose((3, 2, 0, 1))
    elif x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
    if squeeze:
        if in_features and out_features:
            x = x.reshape((out_features, in_features))
        x = np.squeeze(x)
    return torch.from_numpy(np.ascontiguousarray(x))


def load_matconvnet_weights(net: nn.Module, mat_convnet_model_path: str):
    weights = scipy.io.loadmat(mat_convnet_model_path)
    bn_eps = 1e-5
    conv1 = net.backbone.conv1[0]
    conv1.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][0]))
    conv1.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][1]))
    bn1 = net.backbone.conv1[1]
    bn1.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][2]))
    bn1.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][3]))
    mat_bn1_moments = weights['net']['params'][0][0]['value'][0][4]
    bn1.running_mean.data.copy_(weights2tensor(mat_bn1_moments[:, 0]))
    bn1.running_var.data.copy_(weights2tensor((mat_bn1_moments[:, 1] ** 2) - bn_eps))

    conv2 = net.backbone.conv2[0]
    conv2.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][5]))
    conv2.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][6]))
    bn2 = net.backbone.conv2[1]
    bn2.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][7]))
    bn2.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][8]))
    mat_bn2_moments = weights['net']['params'][0][0]['value'][0][9]
    bn2.running_mean.data.copy_(weights2tensor(mat_bn2_moments[:, 0]))
    bn2.running_var.data.copy_(weights2tensor((mat_bn2_moments[:, 1] ** 2) - bn_eps))

    conv3 = net.backbone.conv3[0]
    conv3.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][10]))
    conv3.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][11]))
    bn3 = net.backbone.conv3[1]
    bn3.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][12]))
    bn3.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][13]))
    mat_bn3_moments = weights['net']['params'][0][0]['value'][0][14]
    bn3.running_mean.data.copy_(weights2tensor(mat_bn3_moments[:, 0]))
    bn3.running_var.data.copy_(weights2tensor((mat_bn3_moments[:, 1] ** 2) - bn_eps))

    conv4 = net.backbone.conv4[0]
    conv4.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][15]))
    conv4.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][16]))
    bn4 = net.backbone.conv4[1]
    bn4.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][17]))
    bn4.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][18]))
    mat_bn4_moments = weights['net']['params'][0][0]['value'][0][19]
    bn4.running_mean.data.copy_(weights2tensor(mat_bn4_moments[:, 0]))
    bn4.running_var.data.copy_(weights2tensor((mat_bn4_moments[:, 1] ** 2) - bn_eps))

    conv5 = net.backbone.conv5[0]
    conv5.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][20]))
    conv5.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][21]))

    head = net.head

    if isinstance(head, SiamFCLinearHead):
        head.adjust_gain.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][22]))
        head.adjust_bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][23]))
    elif isinstance(head, SiamFCBNHead):
        head.adjust_bn.weight.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][22]))
        head.adjust_bn.bias.data.copy_(weights2tensor(weights['net']['params'][0][0]['value'][0][23]))
        mat_fin_adjust_bn_moments = weights['net']['params'][0][0]['value'][0][24]

        head.adjust_bn.running_mean.data.copy_(weights2tensor(mat_fin_adjust_bn_moments[:, 0]))
        head.adjust_bn.running_var.data.copy_(weights2tensor((mat_fin_adjust_bn_moments[:, 1] ** 2) - bn_eps))
    else:
        raise Exception('unknown SiamFC head')
