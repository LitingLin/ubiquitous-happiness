import tensorflow as tf
import numpy as np


def tf_resize(img: np.ndarray, output_size):
    return tf.image.resize(img, (output_size[1], output_size[0]))
