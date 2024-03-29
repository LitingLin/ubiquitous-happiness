import numpy as np
import cv2
from data.operator.image.tf.rgb_to_gray import tf_image_rgb_to_gray_keep_channels
from data.operator.bbox.aligned.flip import bbox_horizontal_flip


def blur_augmentation(image):
    def rand_kernel():
        sizes = np.arange(5, 46, 2)
        size = np.random.choice(sizes)
        kernel = np.zeros((size, size))
        c = int(size/2)
        wx = np.random.random()
        kernel[:, c] += 1. / size * wx
        kernel[c, :] += 1. / size * (1-wx)
        return kernel
    kernel = rand_kernel()
    image = cv2.filter2D(image, -1, kernel)
    return image


def gray_augmentation(image):
    return tf_image_rgb_to_gray_keep_channels(image)


def color_augmentation(image, rgb_variance):
    image -= rgb_variance.dot(np.random.randn(3).astype(np.float32))
    image = np.clip(image, 0., 255.)
    return image


def horizontal_flip_augmentation(image, bbox):
    h, w, _ = image.shape
    image = cv2.flip(image, 1)

    return image, bbox_horizontal_flip(bbox, (w, h))
