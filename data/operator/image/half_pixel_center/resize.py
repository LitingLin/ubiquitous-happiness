import tensorflow as tf


def tf_resize(img: tf.Tensor, output_size):
    return tf.image.resize(img, (output_size[1], output_size[0]))
