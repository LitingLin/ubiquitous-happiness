def get_tight_rectangular_bounding_box(bounding_box):
    assert len(bounding_box) % 2 == 0
    odd_index_values = bounding_box[::2]
    even_index_values = bounding_box[1::2]
    x1 = min(odd_index_values)
    x2 = max(odd_index_values)
    y1 = min(even_index_values)
    y2 = max(even_index_values)
    return [x1, y1, x2 - x1, y2 - y1]
