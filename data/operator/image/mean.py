import tensorflow as tf


def tf_get_image_mean(image):
    return tf.math.reduce_mean(image, axis=(0, 1, 2))
