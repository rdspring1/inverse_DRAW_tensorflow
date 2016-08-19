import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer

class LSTM_Encoder():
    def __init__(self, batch_size_, lstm_size_, lstm_layers_, latent_size_):
        self.lstm_size = lstm_size_
        self.lstm_layers = lstm_layers_

        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size_, state_is_tuple=True)
        self.encoder = tf.nn.rnn_cell.MultiRNNCell([self.encoder_cell] * lstm_layers_, True)
        self.state, self.clear_ops = self.create_lstm_state(batch_size_, lstm_size_)

        self.mean_weight = tf.get_variable("mean_QW", [latent_size_, lstm_size_], tf.float32, xavier_init())
        self.mean_bias = tf.get_variable("mean_QB", [latent_size_, 1], tf.float32, tf.constant_initializer()) 

        self.sigma_weight = tf.get_variable("sigma_QW", [latent_size_, lstm_size_], tf.float32, xavier_init())
        self.sigma_bias = tf.get_variable("sigma_QB", [latent_size_, 1], tf.float32, tf.constant_initializer())


    def create_lstm_state(self, batch_size, lstm_size):
        lstm_state = []
        clear_ops = []
        for idx in range(self.lstm_layers):
            c = tf.get_variable("c{0}".format(idx), [batch_size, lstm_size], tf.float32, tf.constant_initializer(), trainable=False)
            h = tf.get_variable("h{0}".format(idx), [batch_size, lstm_size], tf.float32, tf.constant_initializer(), trainable=False)
            lstm_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
            clear_ops.append(c.assign(tf.fill([batch_size, lstm_size], 0.0)))
            clear_ops.append(h.assign(tf.fill([batch_size, lstm_size], 0.0)))
        return lstm_state, clear_ops

    def __call__(self, data, size):
        input_seq = [tf.squeeze(input_, [1]) for input_ in tf.split(1, size, data)]

        with tf.control_dependencies(self.clear_ops):
            outputs, states = tf.nn.rnn(self.encoder, input_seq, self.state)

        encoding = outputs[len(outputs)-1]
        z_mean = tf.matmul(self.mean_weight, encoding, transpose_b=True) + self.mean_bias
        z_log_sigma_sq = tf.matmul(self.sigma_weight, encoding, transpose_b=True) + self.sigma_bias
        return z_mean, z_log_sigma_sq
