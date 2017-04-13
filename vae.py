import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl

np.random.seed(42)
tf.set_random_seed(42)

# Variation Autoencoder (VAE) 
# [1] Tutorial on Variational Autoencoders https://arxiv.org/pdf/1606.05908.pdf
# [2] Auto-Encoding Variational Bayes https://arxiv.org/pdf/1312.6114.pdf
class VariationalAutoencoder(object):

    def __init__(self, sess, s_dim, hidden_dim, learning_rate=0.001):

        self.sess = sess
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.hidden_sample = tf.placeholder(tf.float32, [None, self.hidden_dim])
        
        # Create autoencoder framework
        self.enc_mean, self.enc_log_sigma_sq, self.dec_mean = \
            self.create_networks()
        
        # Define loss function vae cost and reconstruction cost
        self.loss, self.opt = \
            self.create_loss_optimizer()

        # Initialize network parameters
        # By default, xavier initializer is invoked
        self.sess.run(tf.global_variables_initializer())
     
    def create_networks(self):
        
        # Encode mean and variance (log sigma square)
        # in latent space
        enc_mean, enc_log_sigma_sq = \
            self.create_encoder_network()

        # Draw a sample from Gaussian distribution
        eps = tf.random_normal(
            (tf.shape(enc_mean)[0], self.hidden_dim), 0, 1,
            dtype=tf.float32)

        # Reparametrization trick
        # z = mu + sigma * epsilon
        self.hidden_sample = tf.add(
            enc_mean, tf.multiply(tf.sqrt(tf.exp(enc_log_sigma_sq)), eps))

        # Reconstruct the data
        dec_mean = self.create_decoder_network()

        return enc_mean, enc_log_sigma_sq, dec_mean

    def create_encoder_network(self):
        hid_1 = tl.fully_connected(self.inputs, 64, activation_fn=tf.nn.softplus)
        hid_2 = tl.fully_connected(hid_1, 32, activation_fn=tf.nn.softplus)
        output = tl.fully_connected(hid_2, self.hidden_dim * 2, activation_fn=None)
        return output[:, :self.hidden_dim], output[:, -self.hidden_dim:]

    def create_decoder_network(self):
        hid_1 = tl.fully_connected(self.hidden_sample, 64, activation_fn=tf.nn.softplus)
        hid_2 = tl.fully_connected(hid_1, 32, activation_fn=tf.nn.softplus)
        output = tl.fully_connected(hid_2, self.s_dim, activation_fn=None)
        return output

    def create_loss_optimizer(self):
        reconstruct_loss = tf.reduce_sum(
            tf.square(self.inputs - self.dec_mean), 1)

        vae_loss = -0.5 * tf.reduce_sum(
            1 + self.enc_log_sigma_sq -
            tf.square(self.enc_mean) - 
            tf.exp(self.enc_log_sigma_sq), 1)

        loss = tf.reduce_mean(reconstruct_loss + vae_loss)

        opt = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate).minimize(loss)

        return loss, opt

    def train(self, inputs):
        opt, loss = self.sess.run((self.opt, self.loss), 
                                  feed_dict={self.inputs: inputs})
        return loss
    
    def encode(self, inputs):
        return self.sess.run((self.enc_mean, self.enc_log_sigma_sq),
                             feed_dict={self.inputs: inputs})
    
    def generate(self, hidden_sample=None):
        # If hidden_sample is not None, data for this point in latent
        # space is generated. Otherwise, hidden_sample is drawn from
        # prior in latent space.
        if hidden_sample is None:
            hidden_sample = np.random.normal(0, 1, size=[1, self.hidden_dim])
        return self.sess.run(self.dec_mean, feed_dict={
            self.hidden_sample: hidden_sample
        })
    
    def reconstruct(self, inputs):
        return self.sess.run(self.dec_mean, feed_dict={
            self.inputs: inputs
        })
