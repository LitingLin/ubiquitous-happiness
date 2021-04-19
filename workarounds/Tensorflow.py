def silence_tensorflow():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def disable_tensorflow_gpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
