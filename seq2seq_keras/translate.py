"""
This is a Keras implementation of RNN Encoder-Decoder, written for as a final project for CSCI-544.

As we started modifying code from a Tensorflow tutorial, some of the routines such as command line options
and basic file reading routines might resemble the original tutorial code, however, we re-implemented most
of the key parts namely:

1. Our training model is completely re-written with Keras.
2. The WMT dataset is processed with DataFeed, an self sufficient class that handles processing, reading
   and producing batches from the data.
3. A beam search is performed to sample the output probability distribution to get reasonable outcome.

Group members (in alphabetic order):

    Wilka Carvalho
    Yi Ren
    Shuyang Sheng
    Yaning Yu

For questions regarding the project, please leave comments or pull requests to:

    https://github.com/wcarvalho/544-smt-project

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.layers import RepeatVector, Input, merge
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from tensorflow import one_hot
from seq2seq_keras.data_feeder import *

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "wmt", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "wmt", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,  "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("quick_and_dirty", False, "Quick & Dirty settings for fast testing")
tf.app.flags.DEFINE_string("plot_name", None, "base name for plots")
FLAGS = tf.app.flags.FLAGS


def softmax_sampling(x):
    idx = np.random.choice(len(x), p=x)
    ret = np.zeros_like(x)
    ret[idx] = 1
    return ret


def decoding(model, en_input, vocab_size):
    output = []
    prev = np.zeros([vocab_size,1])
    for word in en_input:
        p = model.predict([word, prev])
        cur = softmax_sampling(p)
        prev = cur
        output.append(prev)
    return output


def create_model(vocab_size, en_length, fr_length, hidden_dim):
    en = Input(shape=(en_length,), name='en_input_w')
    s = Embedding(vocab_size, 64, input_length=en_length, mask_zero=True, name='en_embed_s')(en)
    h =LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    

    c = RepeatVector(fr_length, name='repeated_hidden_c')(h)

    fr = Input(shape=(fr_length, vocab_size,), name='fr_input_y')
    decode_input = merge([c, fr], mode='concat',  name='y_cat_c')
    z = LSTM(hidden_dim, return_sequences=True, name='hidden_z')(decode_input)
    p = TimeDistributed(Dense(vocab_size, activation='softmax'), name='prob')(z)
    model = Model(input=[en, fr], output=p)
    if FLAGS.plot_name is not None: 
        plot(model, to_file=FLAGS.plot_name+'.png', show_shapes=True)
    return model


def create_model_test(vocab_size, hidden_dim):
    en = Input(shape=(1,), name='en_input_w')
    s = Embedding(vocab_size, 64, input_length=1, mask_zero=True, name='en_embed_s')(en)
    h =LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    c = RepeatVector(1, name='repeated_hidden_c')(h)

    prev = Input(shape=(1, vocab_size,), name='fr_input_y')
    decode_input = merge([c, prev], mode='concat',  name='y_cat_c')
    z = LSTM(hidden_dim, return_sequences=True, name='hidden_z')(decode_input)
    p = TimeDistributed(Dense(vocab_size, activation='softmax'), name='prob')(z)
    model = Model(input=[en, prev], output=p)
    if FLAGS.plot_name is not None: 
        plot(model, to_file=FLAGS.plot_name+'_test.png', show_shapes=True)
    return model


def one_hot(a, vocab_size):
    return (np.arange(vocab_size) == a[:,:,None]-1).astype(int)


def train():
    # Prepare WMT data
    train_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                                          prefix="giga-fren.release2",
                                          vocab_size=FLAGS.vocab_size,
                                          max_num_samples=FLAGS.max_train_data_size)

    test_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                                         prefix="newstest2013",
                                         vocab_size=FLAGS.vocab_size)

    en_length, fr_length, hidden_dim = 40, 50, 1000
    model_train = create_model(FLAGS.vocab_size, en_length, fr_length, hidden_dim)
    model_test = create_model_test(FLAGS.vocab_size, hidden_dim)
    model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model_test.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model_train.summary())

    for i in range(1000):
        source, target = train_feeder.get_batch(FLAGS.batch_size, en_length=en_length, fr_length=fr_length)
        # embedding handles one hot coding, so source doesn't need it.
        source, target = np.asarray(source), np.asarray(target)
        target = one_hot(target, FLAGS.vocab_size)
        model_train.train_on_batch([source, target], target)
        if i+1 % 100 == 0:
            # TODO: should also run a validation/test
            print("%d iterations" % (i+1))
            model_train.save(FLAGS.train_dir + "/itr_%d.chkpoint" % (i+1), overwrite=False)


if __name__ == "__main__":
    if FLAGS.plot_name is not None: 
        from keras.utils.visualize_util import plot

    if FLAGS.decode:
        decoding()
    else:
        train()
