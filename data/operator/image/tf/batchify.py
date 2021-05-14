import tensorflow as tf


def tf_batchify(tensor):
    return tf.expand_dims(tensor, 0)


def tf_unbatchify(tensor):
    return tf.squeeze(tensor, 0)
