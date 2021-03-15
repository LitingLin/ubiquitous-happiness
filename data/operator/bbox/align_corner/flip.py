def bbox_horizontal_flip(bbox, image_size):
    edge = image_size[0] - 1
    x2 = edge - bbox[0]
    x1 = edge - bbox[2]
    return x1, bbox[1], x2, bbox[3]
