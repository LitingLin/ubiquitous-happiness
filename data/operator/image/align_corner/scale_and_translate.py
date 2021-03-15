from tensorflow.python.ops import gen_image_ops


def align_corners_resize(img, output_size, kernel_type='triangle', antialias=False):
    '''
        For tensorflow v2, image resize functions assume the pixels are locating at (0.5 * x, 0.5 * y),
        which is the half_pixel_centers=True behaviour as in tensorflow v1.
        And pixel mapping function is: o = i * scale + translate
        Since we need corner aligned resizing, top-left pixel (0.5, 0.5) is mapped to (0.5, 0.5),
        bottom-right pixel (i_width - 0.5, i_height - 0.5) is mapped to (o_width - 0.5, o_height - 0.5),
        we have:
            0.5 = 0.5 * scale + translate
            o_size - 0.5 = (i_size - 0.5) * scale + translate
        ==>
            scale = (1 - o_size) / (1 - i_size)
            translate = (1 - scale) / 2
    '''
    n, h, w, c = img.shape()
    assert h != 1 and w != 1
    o_w, o_h = output_size
    sy = (o_h - 1) / (h - 1)
    sx = (o_w - 1) / (w - 1)
    tx = (1 - sx) / 2
    ty = (1 - sy) / 2
    img = gen_image_ops.scale_and_translate(
        img,
        (o_h, o_w),
        (sy, sx),
        (ty, tx),
        kernel_type=kernel_type,
        antialias=antialias)
    return img


def tf_image_scale_and_translate_align_corners(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), kernel_type='triangle', antialias=False):
    '''
        (i_edge_point + 0.5) * tf_scale + tf_translate = o_edge_point + 0.5
            ==> tf_scale = (o_edge_point_1 - o_edge_point_2) / (i_edge_point_1 - i_edge_point_2)
                tf_translate = o_edge_point + 0.5 - (i_edge_point + 0.5) * tf_scale

        (i - input_center) * scale = o - output_center
            ==> i_edge_point_1 = 0, o_edge_point_1 = output_center - input_center * scale
                i_edge_point_2 = image_size - 1, o_edge_point_2 = (image_size - 1 - input_center) * scale + output_center
    ==> tf_scale = scale
        tf_translate = output_center - (0.5 + input_center) * scale + 0.5
    '''
    o_w, o_h = output_size
    ic_x, ic_y = input_center
    oc_x, oc_y = output_center
    s_x, s_y = scale
    t_x = oc_x - (0.5 + ic_x) * s_x + 0.5
    t_y = oc_y - (0.5 + ic_y) * s_y + 0.5
    img = gen_image_ops.scale_and_translate(img, (o_h, o_w), (s_y, s_x), (t_y, t_x), kernel_type=kernel_type, antialias=antialias)
    return img
