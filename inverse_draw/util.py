import functools
import tensorflow as tf
import numpy as np
import numpy.random
import random

def loss_weight(seq_len, max_seq_len):
    weight = np.zeros([1, max_seq_len])
    weight[:, :seq_len] = 1
    return tf.split(1, max_seq_len, tf.constant(weight, dtype=tf.float32))

def copy_generator(num_examples, size, min_seq_len, max_seq_len, SEED=None):
    np.random.seed(SEED)
    np_end_state = np.zeros([1, size])
    np_end_state[:, size-1] = 1

    idx = 0
    while idx < num_examples:
        # Generate Random Sequence
        seq_len = random.randint(min_seq_len, max_seq_len)
        mask = np.random.uniform(0, 1, [seq_len, size]) > 0.5
        seq = mask.astype(float)

        # Input random sequence 
        inputs_ = np.split(seq, seq_len, 0)

        # Sequence End Marker
        inputs_.append(np_end_state)

        # Add Zero placeholders 
        for idx in range(2*max_seq_len - seq_len):
            inputs_.append(np.zeros([1, size]))

        #output_ = np.zeros([2*max_seq_len+1, size])
        #output_[seq_len, size-1] = 1
        #output_[max_seq_len+1:max_seq_len+1+seq_len, :] = seq
        output_ = np.zeros([max_seq_len, size])
        output_[:seq_len, :] = seq

        yield inputs_, output_ 

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
