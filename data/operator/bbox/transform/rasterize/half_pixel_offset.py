def bbox_rasterize_half_pixel_offset(bbox):
    return tuple(int(round(v - 0.5)) for v in bbox)
