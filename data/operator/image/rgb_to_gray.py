import tensorflow as tf


def tf_image_rgb_to_gray_keep_channels(image):
    image = tf.image.rgb_to_grayscale(image)
    return tf.tile(image, [1, 1, 1, 3])
