from tensorflow.python.ops import gen_image_ops


def tf_image_scale_and_translate(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), kernel_type='triangle', antialias=False):
    '''
        i_edge_point * tf_scale + tf_translate = o_edge_point
            ==> tf_scale = (o_edge_point_1 - o_edge_point_2) / (i_edge_point_1 - i_edge_point_2)
                tf_translate = o_edge_point - i_edge_point * tf_scale

        (i - input_center) * scale = o - output_center
            ==> i_edge_point_1 = 0, o_edge_point_1 = output_center - input_center * scale
                i_edge_point_2 = image_size, o_edge_point_2 = (image_size - input_center) * scale + output_center
    ==> tf_scale = scale
        tf_translate = output_center - input_center * scale
    '''
    o_w, o_h = output_size
    ic_x, ic_y = input_center
    oc_x, oc_y = output_center
    s_x, s_y = scale
    t_x = oc_x - ic_x * s_x
    t_y = oc_y - ic_y * s_y
    img = gen_image_ops.scale_and_translate(img, (o_h, o_w), (s_y, s_x), (t_y, t_x), kernel_type=kernel_type, antialias=antialias)
    return img
