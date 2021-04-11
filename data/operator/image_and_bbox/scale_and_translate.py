def tf_scale_and_translate_numpy(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(0, 0, 0), kernel_type='triangle', antialias=False):
    from data.context.coordinate_system import is_using_align_corner_coordinate_system
    if is_using_align_corner_coordinate_system():
        from data.operator.image_and_bbox.align_corner.scale_and_translate import tf_scale_and_translate_numpy
        return tf_scale_and_translate_numpy(img, output_size, scale, input_center, output_center, background_color, kernel_type, antialias)
    else:
        from data.operator.image_and_bbox.half_pixel_center.scale_and_translate import tf_scale_and_translate_numpy
        return tf_scale_and_translate_numpy(img, output_size, scale, input_center, output_center, background_color, kernel_type, antialias)
