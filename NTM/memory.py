import tensorflow as tf

class Memory:
    def __init__(self, name_, mem_dim_, mem_size_):
        self.mem_dim = mem_dim_
        self.mem_size = mem_size_
        self.data = tf.get_variable("memory", [mem_dim_, mem_size_], tf.float32, tf.constant_initializer(1e-6), trainable=False)
        self.clear_op = self.data.assign(tf.fill([mem_dim_, mem_size_], 1e-6))
