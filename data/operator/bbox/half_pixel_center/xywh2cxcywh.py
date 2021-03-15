def bbox_xywh2cxcywh(bbox):
    w = bbox[2]
    h = bbox[3]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    return (cx, cy, w, h)
