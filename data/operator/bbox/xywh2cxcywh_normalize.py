import numpy as np


def bbox_xywh2cxcywh_normalize_(bbox: np.ndarray, image_size):
    bbox = bbox.astype(np.float, copy=False)
    bbox[0] += bbox[2] / 2
    bbox[1] += bbox[1] + bbox[3] / 2

    w, h = image_size
    bbox[0] /= w
    bbox[1] /= h
    bbox[2] /= w
    bbox[3] /= h
    return bbox


def bbox_xywh2cxcywh_normalize(bbox: np.ndarray, image_size):
    return bbox_denormalize_cxcywh2xywh_(bbox.copy(), image_size)


def bbox_denormalize_cxcywh2xywh_(bbox, origin_image_size):
    w, h = origin_image_size

    bbox[0] *= w
    bbox[1] *= h
    bbox[2] *= w
    bbox[3] *= h

    bbox[0] -= bbox[2] /2
    bbox[1] -= bbox[3] /2
    return bbox


def bbox_denormalize_cxcywh2xywh(bbox, origin_image_size):
    return bbox_denormalize_cxcywh2xywh_(bbox.copy(), origin_image_size)
