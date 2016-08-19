import functools
import tensorflow as tf
import numpy as np
import numpy.random
import random

def loss_weight(seq_len, max_seq_len):
    weight = np.zeros([1, max_seq_len])
    weight[:, :seq_len] = 1
    return tf.split(1, max_seq_len, tf.constant(weight, dtype=tf.float32))

def normalize(data):
    return tf.div(data, tf.sqrt(tf.reduce_sum(tf.pow(data, 2.0)) + 1e-10))

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def onehot_initializer(dtype_=tf.float32):
    def _initializer(shape, dtype=dtype_):
        return onehot(shape, dtype) 
    return _initializer

def onehot(shape, dtype=tf.float32):
    max_dim = max(shape)
    min_dim = min(shape)
    return tf.constant(np.eye(max_dim)[:min_dim], dtype)
