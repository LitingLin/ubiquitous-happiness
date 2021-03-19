import numpy as np
import tensorflow as tf
import torch


def image_round_to_uint8(img: np.ndarray):
    return np.round(img).clip(0., 255.).astype(np.uint8)


def tf_image_round_to_uint8(img: np.ndarray):
    img = tf.clip_by_value(img, 0., 255.)
    img = tf.round(img)
    img = tf.cast(img, tf.uint8)
    return img


def torch_image_round_to_uint8(img: torch.Tensor):
    return torch.round(img).clip(0., 255.).to(torch.uint8)
