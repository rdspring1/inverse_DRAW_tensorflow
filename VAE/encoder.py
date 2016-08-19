import tensorflow as tf
import numpy as np
import numpy.random

xavier_init = tf.contrib.layers.xavier_initializer 

class FC_Encoder():
    def __init__(self, data_size_, latent_size_, num_nodes_, num_layers_):
        self.num_layers = num_layers_

        self.weight = [tf.get_variable("Q_W0", [data_size_, num_nodes_], tf.float32, xavier_init())]
        self.bias = [tf.get_variable("Q_B0", [num_nodes_], tf.float32, tf.constant_initializer())]

        for idx in range(1, num_layers_):
            self.weight.append(tf.get_variable("Q_W{0}".format(idx), [num_nodes_, num_nodes_], tf.float32, xavier_init()))
            self.bias.append(tf.get_variable("Q_B{0}".format(idx), [num_nodes_], tf.float32, tf.constant_initializer()))

        self.mean_weight = tf.get_variable("mean_QW", [num_nodes_, latent_size_], tf.float32, xavier_init())
        self.mean_bias = tf.get_variable("mean_QB", [latent_size_], tf.float32, tf.constant_initializer()) 

        self.sigma_weight = tf.get_variable("sigma_QW", [num_nodes_, latent_size_], tf.float32, xavier_init())
        self.sigma_bias = tf.get_variable("sigma_QB", [latent_size_], tf.float32, tf.constant_initializer())

    def __call__(self, data):
        for idx in range(self.num_layers):
            data = tf.nn.relu(tf.matmul(data, self.weight[idx]) + self.bias[idx])

        z_mean = tf.matmul(data, self.mean_weight) + self.mean_bias
        z_log_sigma_sq = tf.matmul(data, self.sigma_weight) + self.sigma_bias
        return z_mean, z_log_sigma_sq
