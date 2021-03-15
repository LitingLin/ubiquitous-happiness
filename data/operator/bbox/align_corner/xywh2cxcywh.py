def bbox_xywh2cxcywh(bbox):
    w = bbox[2] - 1
    h = bbox[3] - 1
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    return (cx, cy, w, h)
