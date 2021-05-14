import tensorflow as tf


def tf_decode_image(path: str, channels=3):
    return tf.io.decode_image(tf.io.read_file(path), channels=channels)


def decode_image(path: str):
    return tf_decode_image(path).numpy()
