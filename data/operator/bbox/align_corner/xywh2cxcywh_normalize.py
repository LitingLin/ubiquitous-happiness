from .xywh2cxcywh import bbox_xywh2cxcywh


def bbox_xywh2cxcywh_normalize(bbox, image_size):
    bbox = bbox_xywh2cxcywh(bbox)

    w, h = image_size
    w = w - 1
    h = h - 1
    return (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
