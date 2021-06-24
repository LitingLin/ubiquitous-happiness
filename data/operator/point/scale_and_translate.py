def xy_point_scale_and_translate(point, scale, input_center, output_center):
    '''
        (i - input_center) * scale = o - output_center
        :return XYXY format
    '''
    x, y = point
    ic_x, ic_y = input_center
    oc_x, oc_y = output_center
    s_x, s_y = scale
    o_x = oc_x + (x - ic_x) * s_x
    o_y = oc_y + (y - ic_y) * s_y
    return (o_x, o_y)
