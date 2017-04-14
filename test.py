import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import vae
import cvae
import input_data

S_DIM = input_data.FUTURE_LEN
C_DIM = input_data.PAST_LEN
HIDDEN_DIM = 10
MODEL = "./results/nn_model_98.ckpt"

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

        all_inputs_mean = []
        all_outputs_mean = []
        all_gen_mean = []

        # plot generating examples
        plot_ex = 6
        while True:
            np.random.shuffle(test_idx)
            fig = plt.figure(figsize=(14, 6))

            gs = gridspec.GridSpec(plot_ex, plot_ex)
            for i in xrange(plot_ex):
                for j in xrange(plot_ex):
                    idx = i * plot_ex + j

                    cond_inputs = test_cond_inputs[
                        test_idx[idx]:test_idx[idx]+1, :]
                    inputs = test_inputs[
                        test_idx[idx]:test_idx[idx]+1, :]

                    ax = plt.subplot(gs[i:i+1, j:j+1])
                    ax.plot(range(C_DIM), cond_inputs[0], 'b')

                    for _ in xrange(20):
                        gen = cond_var_auto_enc.generate(cond_inputs)
                        ax.plot(range(C_DIM, C_DIM + S_DIM), gen[0],
                            'r', alpha=0.5)

                    ax.plot(range(C_DIM, C_DIM + S_DIM), inputs[0], 'b')

            plt.show()


if __name__ == '__main__':
    main()
