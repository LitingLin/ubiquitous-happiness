def bbox_compute_intersection(bbox1, bbox2):
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    return (inter_x1, inter_y1, inter_x2, inter_y2)
