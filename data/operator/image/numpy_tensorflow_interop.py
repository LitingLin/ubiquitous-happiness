import numpy as np
import tensorflow as tf


def image_prepare_tf_operator(image: np.ndarray):
    assert image.ndim == 3
    image = np.expand_dims(image, 0)
    return tf.constant(image)


def image_prepare_tf_operator_as_float32(image: np.ndarray):
    assert image.ndim == 3
    image = np.expand_dims(image, 0)
    return tf.constant(image, dtype=tf.float32)


def image_prepare_tf_operator_batched(image: np.ndarray):
    return tf.constant(image)


def image_end_tf_operator(image: tf.Tensor):
    image = image.numpy()
    assert image.ndim == 4 and image.shape[0] == 1
    return np.squeeze(image, 0)


def image_end_tf_operator_batched(image: tf.Tensor):
    return image.numpy()
