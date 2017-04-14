import numpy as np
import tensorflow as tf
import vae
import cvae
import input_data

S_DIM = 10
C_DIM = 10
HIDDEN_DIM = 10
BATCH_SIZE = 100
TRAIN_EPOCHS = 100
MODEL_SAVE_PATH = './results/'
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

        test_num = int(0.2 * len(all_idx))
        test_cond_inputs = all_cond_inputs[all_idx[-test_num:], :]
        test_inputs = all_inputs[all_idx[-test_num:], :]

        for ep in xrange(TRAIN_EPOCHS):
            
            np.random.shuffle(train_idx)
            steps = int(len(train_idx) / BATCH_SIZE)
            train_loss = 0

            for i in xrange(steps):
                cond_inputs = train_cond_inputs[
                    train_idx[BATCH_SIZE * i : BATCH_SIZE * (i + 1)], :]
                inputs = train_inputs[
                    train_idx[BATCH_SIZE * i : BATCH_SIZE * (i + 1)], :]
                train_loss += cond_var_auto_enc.train(inputs, cond_inputs)

            train_loss /= float(steps)

            _, test_loss = cond_var_auto_enc.reconstruct(
                test_inputs, test_cond_inputs)

            print 'epoch %d train_loss %0.3f test_loss %0.3f' % \
                (ep, train_loss, test_loss)

            save_path = saver.save(sess,
                MODEL_SAVE_PATH + "nn_model_" + str(ep) + ".ckpt")


if __name__ == '__main__':
    main()
