from .siamese import tf_siamese_pair_data_preprocessing


def tf_siamese_with_intermediate_frame_data_preprocessing(data, is_positive):
    '''
        Make sure that data is sorted
    '''
    if len(data) < 3:
        z, z_bbox, x, x_bbox, is_positive = tf_siamese_pair_data_preprocessing(data, is_positive)
        return z, z_bbox, z_bbox, x, x_bbox, is_positive
    else:
        z, z_bbox, x, x_bbox, is_positive = tf_siamese_pair_data_preprocessing([data[0], data[2]], is_positive)
        return z, z_bbox, data[1][1], x, x_bbox, is_positive
