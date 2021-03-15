import tensorflow as tf
import numpy as np


# return dtype=np.float32
def tf_align_corners_resize(img: np.ndarray, output_size):
    output_w, output_h = output_size
    img = tf.image.crop_and_resize(img, ((0, 0, 1, 1),), (0,), (output_h, output_w))
    return img
