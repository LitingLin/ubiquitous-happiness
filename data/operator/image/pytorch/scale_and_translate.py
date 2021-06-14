import torchvision
import torchvision.transforms.functional


def torch_image_scale_and_translate_align_corners(img, output_size, scale, input_center, output_center, background_color=None):
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
    assert s_x == s_y

    return torchvision.transforms.functional.affine(img, 0, (t_x, t_y), s_x, (0, 0), torchvision.transforms.functional.InterpolationMode.BICUBIC, fill=background_color)
