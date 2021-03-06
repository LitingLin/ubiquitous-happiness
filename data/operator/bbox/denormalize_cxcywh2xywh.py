import copy
from .cxcywh2xywh import bbox_cxcywh2xywh


def bbox_denormalize_cxcywh2xywh(bbox, origin_image_size):
    w, h = origin_image_size

    w = w - 1
    h = h - 1

    bbox = copy.copy(bbox)
    bbox[0] *= w
    bbox[1] *= h
    bbox[2] *= w
    bbox[3] *= h

    return bbox_cxcywh2xywh(bbox)
