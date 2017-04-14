import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vae
import cvae
import input_data

S_DIM = 10
C_DIM = 10
HIDDEN_DIM = 10
MODEL = "./nn_model.ckpt"

def main():

    np.random.seed(42)

    with tf.Session() as sess:
        cond_var_auto_enc = cvae.ConditionalVariationalAutoencoder(
            sess, S_DIM, C_DIM, HIDDEN_DIM)
        saver = tf.train.Saver()
        saver.restore(sess, MODEL)

        all_cond_inputs, all_inputs = input_data.read_data()
        all_idx = range(len(all_inputs))
        np.random.shuffle(all_idx)

        test_num = int(0.2 * len(all_idx))
        test_cond_inputs = all_cond_inputs[all_idx[-test_num:], :]
        test_inputs = all_inputs[all_idx[-test_num:], :]
        test_idx = range(len(test_inputs))
        np.random.shuffle(test_idx)

        all_inputs_mean = []
        all_outputs_mean = []
        all_gen_mean = []

        for i in xrange(1000):

            cond_inputs = test_cond_inputs[test_idx[i]:test_idx[i]+1, :]
            inputs = test_inputs[test_idx[i]:test_idx[i]+1, :]

            plt.plot(range(10), cond_inputs[0])
            for _ in xrange(20):
                gen = cond_var_auto_enc.generate(cond_inputs)
                plt.plot(range(10, 20), gen[0], 'r', alpha=0.5)
            plt.plot(range(10, 20), inputs[0], 'b')
            plt.title(test_idx[i])
            plt.show()


if __name__ == '__main__':
    main()
