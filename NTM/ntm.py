import tensorflow as tf
from heads import WriteHead, ReadHead

class NTM:
    def __init__(self, scope_, controller_, read_heads_, write_heads_, num_nodes_, size_):
        self.controller = controller_
        self.read_heads = read_heads_
        self.write_heads = write_heads_
        self.w_output = tf.get_variable("w_output", [num_nodes_, size_], tf.float32, tf.contrib.layers.xavier_initializer())
        self.b_output = tf.get_variable("b_output", [size_], tf.float32, tf.constant_initializer(1e-6))

    def step(self, input_, state, memory):
        with tf.name_scope('ntm_step') as scope:
            read_list = []
            for head in self.read_heads:
                read_list.append(head.generate_weight(memory, state))
            read_t = ReadHead.read(tf.concat(0, read_list), memory)

            next_state = self.controller.step(input_, read_t)

            erase_list = []
            write_list = []
            for head in self.write_heads:
                erase, write = head.build_weights(memory, state)
                erase_list.append(erase)
                write_list.append(write)
            erase_ops = tf.pack(erase_list)
            write_ops = tf.pack(write_list)

            memory_tilde = WriteHead.apply_erase(memory, erase_ops)
            new_memory = WriteHead.apply_write(memory_tilde, write_ops)
            return next_state, new_memory

    def execute(self, state_list):
        with tf.name_scope('ntm_output') as scope:
            states = tf.concat(0, state_list)
            return tf.matmul(states, self.w_output) + self.b_output
