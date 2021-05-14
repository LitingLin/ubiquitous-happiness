def tf_resize(img, output_size):
    from data.context.coordinate_system import is_using_align_corner_coordinate_system
    if is_using_align_corner_coordinate_system():
        from data.operator.image.align_corner.resize import tf_resize
        return tf_resize(img, output_size)
    else:
        from data.operator.image.half_pixel_center.resize import tf_resize
        return tf_resize(img, output_size)
