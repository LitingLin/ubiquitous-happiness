import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
import numpy as np


# return dtype=np.float32
def aligned_corner_resize(img: np.ndarray, output_size):
    assert img.dtype == np.uint8
    assert len(img.shape) == 3 and img.shape[2] == 3
    input_h, input_w, input_c = img.shape
    output_w, output_h = output_size
    img = tf.constant(img.reshape((1, input_h, input_w, input_c)), dtype=tf.uint8)
    sy = (output_h - 1) / (input_h - 1)
    sx = (output_w - 1) / (input_w - 1)
    # scale = [sy, sx]
    tx = (1 - sx) / 2
    ty = (1 - sy) / 2
    # translate = [ty, tx]
    img = gen_image_ops.scale_and_translate(
        img,
        (output_h, output_w),
        (sy, sx),
        (ty, tx),
        kernel_type='triangle',
        antialias=False)
    img = img.numpy()
    img = img.reshape(output_h, output_w, input_c)
    return img
