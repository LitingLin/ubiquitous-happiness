def get_center_point_from_wh(size):
    return size[0] / 2, size[1] / 2


def get_center_point_from_xywh(bbox):
    return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)
