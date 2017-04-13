import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl

np.random.seed(42)
tf.set_random_seed(42)

# Variation Autoencoder (VAE) 
# [1] Tutorial on Variational Autoencoders https://arxiv.org/pdf/1606.05908.pdf
# [2] Auto-Encoding Variational Bayes https://arxiv.org/pdf/1312.6114.pdf
class VariationalAutoencoder(object):

    def __init__(self, sess, s_dim, hidden_dim=4, 
                 learning_rate=0.001, batch_size=100):

        self.sess = sess
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.input = tf.placeholder(tf.float32, [None, self.s_dim])
        
        # Create autoencoder framework
        self.create_networks()
        
        # Define loss function vae cost and reconstruction cost
        self.create_loss_optimizer()
     
    def create_networks(self):
        
        # Encode mean and variance (log sigma square)
        # in latent space
        self.enc_mean, self.enc_log_sigma_sq = \
            self.create_encoder_network()

        # Draw a sample from Gaussian distribution
        self.eps = tf.random_normal(
            (self.batch_size, self.hidden_dim), 0, 1, 
            dtype=tf.float32)

        # Reparametrization trick
        # z = mu + sigma * epsilon
        self.hidden_sample = tf.add(
            self.enc_mean, 
            tf.multiply(tf.sqrt(tf.exp(self.enc_log_sigma_sq)), self.eps))

        # Reconstruct the data
        self.dec_mean = self.create_decoder_network()

    def create_encoder_network(self):
        hid_1 = tl.fully_connected(self.inputs, 32, activation_fn=tf.nn.softplus)
        hid_2 = tl.fully_connected(hid_1, 16, activation_fn=tf.nn.softplus)
        output = tl.fully_connected(hid_2, self.hidden_dim * 2, activation_fn=None)
        return output[:, self.hidden_dim:], output[:, -self.hidden_dim:]

    def create_decoder_network(self):
        hid_1 = tl.fully_connected(self.hidden_sample, 32, activation_fn=tf.nn.softplus)
        hid_2 = tf.fully_connected(hid_1, 16, activation_fn=tf.nn.softplus)
        output = tl.fully_connected(hid_2, 1, activation_fn=None)
        return output

    def create_loss_optimizer(self):
        reconstruct_loss = tf.reduce_mean(
            tf.square(self.input - self.dec_mean), 1)

        vae_loss = -0.5 * tf.reduce_mean(
            1 + self.enc_log_sigma_sq -
            tf.square(self.enc_mean) - 
            tf.exp(self.enc_log_sigma_sq), 1)

        self.loss = reconstruct_loss + vae_loss

        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, inputs):
        opt, loss = self.sess.run((self.opt, self.loss), 
                                  feed_dict={self.inputs: inputs})
        return loss
    
    def encode(self, inputs):
        return self.sess.run(self.enc_mean, feed_dict={
            self.inputs: inputs
        })
    
    def generate(self, hidden_sample=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if hidden_sample is None:
            hidden_sample = np.random.normal(
            	size=[1, self.hidden_dim])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.dec_mean, feed_dict={
            self.hidden_sample: hidden_sample
        })
    
    def reconstruct(self, inputs):
        return self.sess.run(self.dec_mean, feed_dict={
            self.inputs: inputs
        })
