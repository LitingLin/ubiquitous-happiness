import tensorflow as tf


def tf_image_gray_to_rgb_if_is_gray(tensor: tf.Tensor):
    if tensor.shape[-1] == 1:
        return tf.image.grayscale_to_rgb(tensor)
    elif tensor.shape[-1] == 3:
        return tensor
    else:
        raise NotImplementedError
