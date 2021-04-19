import tensorflow as tf


def tf_batchify(tensor):
    return tf.expand_dims(tensor, 0)


def tf_unbatchify(tensor):
    return tf.squeeze(tensor, 0)


def unbatchify(tensor):
    return tensor.squeeze()


def torch_batchify(tensor):
    return tensor.unsqueeze(0)
