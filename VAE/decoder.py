import tensorflow as tf
import numpy as np
import numpy.random

xavier_init = tf.contrib.layers.xavier_initializer 

class FC_Decoder():
    def __init__(self, batch_size_, data_size_, latent_size_, num_nodes_, num_layers_):
        self.num_layers = num_layers_
        self.batch_size = batch_size_
        self.latent_size = latent_size_

        self.weight = [tf.get_variable("P_W0", [latent_size_, num_nodes_], tf.float32, xavier_init())]
        self.bias = [tf.get_variable("P_B0", [num_nodes_], tf.float32, tf.constant_initializer())]

        for idx in range(1, num_layers_):
            self.weight.append(tf.get_variable("P_W{0}".format(idx), [num_nodes_, num_nodes_], tf.float32, \
                xavier_init()))
            self.bias.append(tf.get_variable("P_B{0}".format(idx), [num_nodes_], tf.float32, tf.constant_initializer()))

        self.vae_weight = tf.get_variable("vae_PW", [num_nodes_, data_size_], tf.float32, xavier_init())
        self.vae_bias = tf.get_variable("vae_PB", [data_size_], tf.float32, tf.constant_initializer()) 

    def __call__(self, z_mean, z_log_sigma_sq):
        epsilon = tf.random_normal([self.batch_size, self.latent_size])
        data = z_mean + tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), epsilon)
        return self.generate(data)

    def generate(self, data):
        for idx in range(self.num_layers):
            data = tf.nn.relu(tf.matmul(data, self.weight[idx]) + self.bias[idx])

        data_hat = tf.sigmoid(tf.matmul(data, self.vae_weight) + self.vae_bias)
        return data_hat 
