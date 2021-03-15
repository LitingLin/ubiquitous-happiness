def is_bbox_validity(bbox):
    return bbox[0] <= bbox[2] and bbox[1] <= bbox[3]

