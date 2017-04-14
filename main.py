import numpy as np
import tensorflow as tf
import vae
import cvae
import input_data

S_DIM = 10
C_DIM = 10
HIDDEN_DIM = 10
BATCH_SIZE = 100
TRAIN_EPOCHS = 100000
MODEL = None

def main():

    np.random.seed(42)

    with tf.Session() as sess:
        cond_var_auto_enc = cvae.ConditionalVariationalAutoencoder(
            sess, S_DIM, C_DIM, HIDDEN_DIM)
        saver = tf.train.Saver()

        all_cond_inputs, all_inputs = input_data.read_data()
        data_idx = range(len(all_inputs))

        for ep in xrange(TRAIN_EPOCHS):
            
            np.random.shuffle(data_idx)
            cond_inputs = all_cond_inputs[data_idx[:BATCH_SIZE], :]
            inputs = all_inputs[data_idx[:BATCH_SIZE], :]

            loss = cond_var_auto_enc.train(inputs, cond_inputs)
            print 'epoch %d loss %0.3f\r' % (ep, loss),

        save_path = saver.save(sess, "./nn_model.ckpt")


if __name__ == '__main__':
    main()
