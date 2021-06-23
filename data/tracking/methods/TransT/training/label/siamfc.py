import torch
import numpy as np


def create_label(size, r_pos, r_neg, total_stride):
    def logistic_label(x, y, r_pos, r_neg):
        dist = np.abs(x) + np.abs(y)  # block distance
        label = np.where(dist <= r_pos,
                          np.ones_like(x),
                          np.where(dist < r_neg,
                                   np.ones_like(x) * 0.5,
                                   np.zeros_like(x)))
        return label

    # distances along x- and y-axis
    c, h, w = size
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    x, y = np.meshgrid(x, y)

    # create logistic labels
    r_pos = r_pos / total_stride
    r_neg = r_neg / total_stride
    label = logistic_label(x, y, r_pos, r_neg)

    # repeat to size
    label = label.reshape((1, h, w))
    label = np.tile(label, (c, 1, 1))

    # convert to tensors
    return torch.from_numpy(label).float()


def create_neg_label(size):
    return torch.zeros(size, dtype=torch.float)


class SiamFCLabelGenerator:
    def __init__(self, size, r_pos, r_neg, total_stride):
        self.size = [1] + size
        self.r_pos = r_pos
        self.r_neg = r_neg
        self.total_stride = total_stride

    def __call__(self, _, is_positive):
        if is_positive:
            return create_label(self.size, self.r_pos, self.r_neg, self.total_stride)
        else:
            return create_neg_label(self.size)
