import math


def bbox_xyxy_rasterize_in_open_interval(bbox):
    x1 = math.floor(bbox[0])
    if x1 == bbox[0]:
        x1 -= 1
    y1 = math.floor(bbox[1])
    if y1 == bbox[1]:
        y1 -= 1
    x2 = math.ceil(bbox[2])
    if x2 == bbox[2]:
        x2 += 1
    y2 = math.ceil(bbox[3])
    if y2 == bbox[3]:
        y2 += 1
    return [x1, y1, x2, y2]
