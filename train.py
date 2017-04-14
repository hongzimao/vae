import numpy as np
import tensorflow as tf
import vae
import cvae
import input_data

S_DIM = 10
C_DIM = 10
HIDDEN_DIM = 10
BATCH_SIZE = 100
TRAIN_EPOCHS = 50000
MODEL = None

def main():

    np.random.seed(42)

    with tf.Session() as sess:
        cond_var_auto_enc = cvae.ConditionalVariationalAutoencoder(
            sess, S_DIM, C_DIM, HIDDEN_DIM)
        saver = tf.train.Saver()

        all_cond_inputs, all_inputs = input_data.read_data()
        all_idx = range(len(all_inputs))
        np.random.shuffle(all_idx)

        train_num = int(0.8 * len(all_idx))
        train_cond_inputs = all_cond_inputs[all_idx[:train_num], :]
        train_inputs = all_inputs[all_idx[:train_num], :]
        train_idx = range(len(train_inputs))

        for ep in xrange(TRAIN_EPOCHS):
            
            np.random.shuffle(train_idx)
            cond_inputs = train_cond_inputs[train_idx[:BATCH_SIZE], :]
            inputs = train_inputs[train_idx[:BATCH_SIZE], :]

            loss = cond_var_auto_enc.train(inputs, cond_inputs)
            print 'epoch %d loss %0.3f\r' % (ep, loss),

        save_path = saver.save(sess, "./nn_model.ckpt")


if __name__ == '__main__':
    main()
