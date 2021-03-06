import numpy as np


def bbox_cxcywh2xywh(bbox):
    w_2 = bbox[2] / 2
    h_2 = bbox[3] / 2
    x = bbox[0] - w_2
    y = bbox[1] - h_2
    w = bbox[2] + 1
    h = bbox[3] + 1
    return np.array([x, y, w, h])
