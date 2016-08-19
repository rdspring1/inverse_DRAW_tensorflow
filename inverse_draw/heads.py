import tensorflow as tf
import util

# memory shape = (N memory_size, M memory dimension)
# ctrl_state = (batch_size, c ctrl_size)
# weight = (N memory_size, batch_size)
# head_output = (batch_size, M memory dimension)
class Head:
    def __init__(self, name_, controller_, num_shifts_, mem_dim_, mem_size_, ctrl_size_, batch_size_):
        self.controller = controller_
        self.mem_dim = mem_dim_
        self.mem_size = mem_size_
        self.ctrl_size = ctrl_size_
        self.num_shift = num_shifts_
        self.batch_size = batch_size_

        with tf.variable_scope(name_):
            self.weight = tf.get_variable("weight", [batch_size_, mem_size_], tf.float32, util.onehot_initializer(), trainable=False)
            self.clear_op = self.weight.assign(util.onehot([batch_size_, mem_size_]))

            self.w_key = tf.get_variable("w_key", [ctrl_size_, mem_dim_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_key = tf.get_variable("b_key", [mem_dim_], tf.float32, tf.constant_initializer())
        
            self.w_beta = tf.get_variable("w_beta", [ctrl_size_, 1], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_beta = tf.get_variable("b_beta", [1], tf.float32, tf.constant_initializer())
        
            self.w_gate = tf.get_variable("w_gate", [ctrl_size_, 1], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_gate = tf.get_variable("b_gate", [1], tf.float32, tf.constant_initializer())

            self.w_shift = tf.get_variable("w_shift", [ctrl_size_, num_shifts_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_shift = tf.get_variable("b_shift", [num_shifts_], tf.float32, tf.constant_initializer())

            self.w_gamma = tf.get_variable("w_gamma", [ctrl_size_, 1], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_gamma = tf.get_variable("b_gamma", [1], tf.float32, tf.constant_initializer())

    # ctrl_state = (batch_size, ctrl_size)
    # k = (batch_size, M memory dimension)
    # memory shape = (M memory dimension, N memory_size)
    # weight vector = (batch_size, N memory_size)
    def cosine_simularity(self, k, memory):
        return tf.matmul(util.normalize(k), util.normalize(memory))

    def generate_weight(self, memory, ctrl_state):
        self.key_t = tf.nn.relu((tf.matmul(ctrl_state, self.w_key) + self.b_key))
        self.beta_t = tf.nn.relu(tf.matmul(ctrl_state, self.w_beta) + self.b_beta)
        self.gate_t = tf.nn.sigmoid(tf.matmul(ctrl_state, self.w_gate) + self.b_gate) 
        self.shift_t = tf.nn.softmax(tf.matmul(ctrl_state, self.w_shift) + self.b_shift) 
        self.gamma_t = tf.nn.relu(tf.matmul(ctrl_state, self.w_gamma) + self.b_gamma) + 1.0

        with tf.name_scope('content_addressing') as scope:
            self.content_address = tf.nn.softmax(tf.mul(self.beta_t, self.cosine_simularity(self.key_t, memory)))
        
        with tf.name_scope('interpolation') as scope:
            self.interpolate = tf.mul(self.gate_t, self.content_address) + tf.mul(1.0 - self.gate_t, self.weight)

        # Convolution Shift
        # shift - batch_size, num_shifts
        # weight - batch_size, memory_size
        # input - batch_size, height, width, channel
        # filter - height, width, in_channel, out_channel
        with tf.name_scope('conv_shift') as scope:
            self.shift_t = tf.expand_dims(self.shift_t, -1)
            self.shift_t = tf.expand_dims(self.shift_t, -1)
            self.shift_t = tf.transpose(self.shift_t, [1, 2, 3, 0])

            self.interpolate = tf.expand_dims(self.interpolate, -1)
            self.interpolate = tf.expand_dims(self.interpolate, -1)

            conv_shift = tf.nn.conv2d(self.interpolate, self.shift_t, [1,1,1,1], 'SAME')
            self.w_tilde = tf.squeeze(conv_shift, [2,3])
        
        with tf.name_scope('sharpening') as scope:
            self.new_weight = tf.pow(self.w_tilde + 1e-10, self.gamma_t)
            return self.weight.assign(tf.truediv(self.new_weight, tf.reduce_sum(self.new_weight)))

class WriteHead(Head):
    def __init__(self, name_, controller_, num_shifts_, mem_dim_, mem_size_, ctrl_size_, batch_size_):
        Head.__init__(self, name_, controller_, num_shifts_, mem_dim_, mem_size_, ctrl_size_, batch_size_)
        with tf.variable_scope(name_):
            self.w_erase = tf.get_variable("w_erase", [ctrl_size_, mem_dim_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_erase = tf.get_variable("b_erase", [mem_dim_], tf.float32, tf.constant_initializer(0))
        
            self.w_write = tf.get_variable("w_write", [ctrl_size_, mem_dim_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_write = tf.get_variable("b_write", [mem_dim_], tf.float32, tf.constant_initializer(0))

    def build_weights(self, memory, ctrl_state):
        with tf.name_scope('erase_write_weights') as scope:
            new_weight = self.generate_weight(memory, ctrl_state)
            self.erase_vector = tf.nn.sigmoid(tf.matmul(ctrl_state, self.w_erase) + self.b_erase)
            self.write_vector = tf.nn.relu(tf.matmul(ctrl_state, self.w_write) + self.b_write)
            erase_op = tf.matmul(self.erase_vector, 1.0 - new_weight, transpose_a=True)
            write_op = tf.matmul(self.write_vector, new_weight, transpose_a=True)
            return erase_op, write_op

    @staticmethod
    def apply_erase(memory, erase_ops):
        with tf.name_scope('erase_update') as scope:
            return tf.foldl(lambda a, x: tf.mul(a, x), erase_ops, memory, name="memory_tilde")

    @staticmethod
    def apply_write(memory_tilde, write_ops):
        with tf.name_scope('write_update') as scope:
            return tf.foldl(lambda a, x: tf.add(a, x), write_ops, memory_tilde, name="new_memory")

class ReadHead(Head):
    def __init__(self, name_, controller_, num_shifts_, mem_dim_, mem_size_, ctrl_size_, batch_size_):
        Head.__init__(self, name_, controller_, num_shifts_, mem_dim_, mem_size_, ctrl_size_, batch_size_)

    # w - batch size x memory size
    # memory - memory_dim x memory size
    # result - batch size x memory_dim
    @staticmethod
    def read(weight, memory):
        with tf.name_scope('read') as scope:
            return tf.matmul(weight, memory, transpose_b=True)
