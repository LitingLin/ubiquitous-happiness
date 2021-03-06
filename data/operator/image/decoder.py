import tensorflow as tf


def decode_image(path: str):
    return tf.io.decode_image(tf.io.read_file(path)).numpy()
