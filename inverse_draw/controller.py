import tensorflow as tf

# data - (batchSize x feature dimension)
# read - (batchSize x memory dimension)
# memory - (memory dimension x memory size)
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

    def step(self, read):
        if self.time_step > 0:
            tf.get_variable_scope().reuse_variables()
        self.time_step += 1
 
        with tf.name_scope('ctrl_step') as scope:
             output, state = self.lstm(read, self.lstm_state)
             with tf.control_dependencies(self.assign_lstm_state(state)):
                 return tf.identity(output, name="ctrl_lstm_output")
