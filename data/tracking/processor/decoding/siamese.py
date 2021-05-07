from data.operator.image.decoder import tf_decode_image


def tf_siamese_pair_data_preprocessing(data, is_positive):
    data = [(tf_decode_image(image_path), bounding_box) for (image_path, bounding_box) in data]
    if len(data) == 1:
        return data[0][0], data[0][1], data[0][0], data[0][1], is_positive
    else:
        return data[0][0], data[0][1], data[1][0], data[1][1], is_positive
