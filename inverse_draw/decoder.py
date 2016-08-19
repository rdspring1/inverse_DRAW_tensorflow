import tensorflow as tf
from heads import WriteHead, ReadHead

xavier_init = tf.contrib.layers.xavier_initializer

class NTM_Decoder():
    def __init__(self, controller_, read_heads_, num_nodes_, size_):
        self.controller = controller_
        self.read_heads = read_heads_
        self.w_output = tf.get_variable("w_output", [num_nodes_, size_], tf.float32, tf.contrib.layers.xavier_initializer())
        self.b_output = tf.get_variable("b_output", [size_], tf.float32, tf.constant_initializer(1e-6))

    def generate_memory(self, z_mean, z_log_sigma_sq, epsilon):
        return tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), epsilon), name="memory")

    def step(self, state, memory):
        with tf.name_scope('ntm_step') as scope:
            read_list = []
            for head in self.read_heads:
                read_list.append(head.generate_weight(memory, state))
            read_matrix = ReadHead.read(tf.concat(0, read_list), memory)
            read_t = tf.reshape(read_matrix, [1, -1])
            return self.controller.step(read_t)

    def execute(self, state_list):
        with tf.name_scope('ntm_output') as scope:
            states = tf.concat(0, state_list)
            return tf.matmul(states, self.w_output) + self.b_output
