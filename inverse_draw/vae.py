import tensorflow as tf
from util import lazy_property

class NTM_VAE():
    def __init__(self, encoder_network_, decoder_network_, mem_dim_, mem_size_, batch_size_, ntm_size_):
        self.encoder_network = encoder_network_
        self.decoder_network = decoder_network_
        self.mem_dim = mem_dim_
        self.mem_size = mem_size_
        self.batch_size = batch_size_
        self.ntm_size = ntm_size_

    def generate(self, read_only_memory, size):
        ntm_state = tf.constant(1.0, tf.float32, [self.batch_size, self.ntm_size], "ntm_state")
        state_list = []
        for idx in range(size):
            state_list.append(self.decoder_network.step(ntm_state, read_only_memory))
        return self.decoder_network.execute(state_list)

    def inference(self, data, size):
        z_mean, z_log_sigma_sq = self.encoder_network(data, size)
        epsilon = tf.random_normal([self.mem_dim, self.mem_size], name="epsilon")
        read_only_memory = self.decoder_network.generate_memory(z_mean, z_log_sigma_sq, epsilon)
        return self.generate(read_only_memory, size), z_mean, z_log_sigma_sq

    def loss_func(self, data, size):
        data_hat, z_mean, z_log_sigma_sq = self.inference(data, size)
        reconstruction_loss = -tf.reduce_sum(data * tf.log(data_hat + 1e-6) + (1-data) * tf.log(1-data_hat + 1e-6), 0)
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 0)
        return tf.reduce_mean(reconstruction_loss + latent_loss)

    def latent(self, data, size):
        z_mean, z_log_sigma_sq = self.encoder_network(data, size)
        return z_mean

    def random_generate(self, size):
        read_only_memory = tf.random_normal([self.mem_dim, self.mem_size], name="epsilon")
        return self.generate(read_only_memory, size)
