import tensorflow as tf
from util import lazy_property

class VAE():
    def __init__(self, encoder_network_, decoder_network_):
        self.encoder_network = encoder_network_
        self.decoder_network = decoder_network_
        self.loss_func

    def loss_func(self, data):
        z_mean, z_log_sigma_sq = self.encoder_network(data)
        data_hat = self.decoder_network(z_mean, z_log_sigma_sq)

        reconstruction_loss = -tf.reduce_sum(data * tf.log(data_hat + 1e-6) + (1-data) * tf.log(1-data_hat + 1e-6), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
        return tf.reduce_mean(reconstruction_loss + latent_loss)

    def latent(self, data):
        z_mean, z_log_sigma_sq = self.encoder_network(data)
        return z_mean

    def reconstruct(self, data):
        z_mean, z_log_sigma_sq = self.encoder_network(data)
        return self.decoder_network(z_mean, z_log_sigma_sq)

    def generate(self, z_mean, batch_size, image_size, channels):
        data_hat = self.decoder_network.generate(z_mean)
        return tf.reshape(data_hat, [batch_size, image_size, image_size, channels])

    def random_generate(self, batch_size, latent_size, image_size, channels):
        z_mean = tf.random_normal([batch_size, latent_size])
        return self.generate(z_mean, batch_size, image_size, channels)
