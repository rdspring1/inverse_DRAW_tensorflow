import tensorflow as tf

# data - (batchSize x feature dimension)
# read - (batchSize x memory dimension)
# memory - (memory dimension x memory size)
class FC_Controller():
    def __init__(self, scope_, read_heads_, mem_dim_, input_, output_):
        with tf.variable_scope(scope_):
            self.w_input = tf.get_variable("w_input", [input_, output_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.w_read = tf.get_variable("w_read", [read_heads_ * mem_dim_, output_], tf.float32, tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable("bias", [output_], tf.float32, tf.constant_initializer())

    def step(self, data, read):
        with tf.name_scope('ctrl_step') as scope:
            return tf.nn.relu(tf.matmul(data, self.w_input) + tf.matmul(tf.reshape(read, [1, -1]), self.w_read) + self.bias)

class RNN_Controller():
    def __init__(self, scope_, mem_dim_, lstm_size_, num_layers_, batch_size_):
        self.time_step = 0
        self.num_layers = num_layers_
        with tf.variable_scope(scope_):
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size_, state_is_tuple=True)
            self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * num_layers_, True)
            self.lstm_state, self.clear_ops = self.create_lstm_state(batch_size_, lstm_size_)

    def create_lstm_state(self, batch_size, lstm_size):
        lstm_state = []
        clear_ops = []
        for idx in range(self.num_layers):
            c = tf.get_variable("c{0}".format(idx), [batch_size, lstm_size], tf.float32, tf.constant_initializer(), trainable=False)
            h = tf.get_variable("h{0}".format(idx), [batch_size, lstm_size], tf.float32, tf.constant_initializer(), trainable=False)
            lstm_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
            clear_ops.append(c.assign(tf.fill([batch_size, lstm_size], 0.0)))
            clear_ops.append(h.assign(tf.fill([batch_size, lstm_size], 0.0)))
        return lstm_state, clear_ops

    def assign_lstm_state(self, values):
        assignment = []
        for idx in range(self.num_layers):
            assignment.append(self.lstm_state[idx].c.assign(values[idx].c))
            assignment.append(self.lstm_state[idx].h.assign(values[idx].h))
        return assignment

    def step(self, data, read):
        if self.time_step > 0:
            tf.get_variable_scope().reuse_variables()
        self.time_step += 1

        with tf.name_scope('ctrl_step') as scope:
             input_ = tf.concat(1, [data, tf.reshape(read, [1, -1])])
             output, state = self.lstm(input_, self.lstm_state)
             with tf.control_dependencies(self.assign_lstm_state(state)):
                 return tf.identity(output, name="ctrl_lstm_output")
