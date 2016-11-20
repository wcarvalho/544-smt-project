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

import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import RepeatVector, Input, merge
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from tensorflow import one_hot

# FIXME: this is sloppy, whoever did this
from data_feeder import *
from tester import SMT_Tester

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("embedding_size", 64, "Size of word embedding")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "wmt", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "wmt", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,  "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_string("plot_name", None, "base name for plots")
FLAGS = tf.app.flags.FLAGS


def softmax_sampling(x):
    idx = np.random.choice(len(x), p=x)
    ret = np.zeros_like(x)
    ret[idx] = 1
    return ret


def create_model(vocab_size, en_length, fr_length, hidden_dim):
    en = Input(shape=(en_length,), name='en_input_w')
    s = Embedding(vocab_size, FLAGS.embedding_size, input_length=en_length, mask_zero=True, name='en_embed_s')(en)
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
    s = Embedding(vocab_size, FLAGS.embedding_size, input_length=1, mask_zero=True, name='en_embed_s')(en)
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

    # TODO: should call test_feeder.get_batch() once to get the entire test set, which is reasonably small.

    en_length, fr_length, hidden_dim = 40, 50, 1000
    model_train = create_model(FLAGS.vocab_size, en_length, fr_length, hidden_dim)
    model_test = create_model_test(FLAGS.vocab_size, hidden_dim)
    model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model_test.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model_train.summary())

    niterations = 10000
    frequency = 100
    for i in range(niterations):
        source, target = train_feeder.get_batch(FLAGS.batch_size, en_length=en_length, fr_length=fr_length)
        source, target = np.asarray(source), np.asarray(target)
        target = one_hot(target, FLAGS.vocab_size) # embedding handles one hot coding, so source doesn't need it.
        loss = model_train.train_on_batch([source, target], target)
        print("Iteration: %d | Loss = %.3f" % (i+1, loss))

        if (i+1) % frequency == 0:
            # TODO: should also run a validation/test
            # 1. somehow copy weights from train model to test model
            # 2. model_test.test_on_batch(...)
            
            # do_validation
            # do_testing

            # optional: print("saving a model...")
            model_train.save(os.path.join(FLAGS.train_dir, "itr_%d.chkpoint" % (i+1)), overwrite=False)

    # use best weights for testing


def test():
    vocab_size = FLAGS.vocab_size
    embedding_size = FLAGS.embedding_size

    test_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                                         prefix="newstest2013",
                                         vocab_size=vocab_size)

    # FIXME: make below flags
    en_length, hidden_dim = 40, 1000
    tester = SMT_Tester(en_length, hidden_dim, vocab_size, vocab_size, embedding_size)
    # tester.load_weights()

    for i in range(10):
        en_sentence, _ = test_feeder.get_batch(1, en_length=en_length)
        en_sentence = np.array(en_sentence)
        tester.encode(en_sentence)
        ## Beam Search
        source_vector = tester.decode()
        # initialize matrixs and vectors
        final_translation_index_list = []
        index_currentw_matrix = np.zeros((50, 50)) 
        index_previousw_matrix = np.zeros((50, 50))
        product_vector = np.zeros((1, 50 * vocab_size))
        # sort the source_vector and get the top 50 largest probabilities
        fifty_index = np.argsort(source_vector, kind = 'heapsort')[:, -50:]
        fifty_largest = np.sort(source_vector, kind = 'heapsort')[:, -50:]
        index_currentw_matrix[0, :] = fifty_index
        # generate the matrix of top 50 French words for each English word
        for j in range(1, 50):
            list_of_fifty_vector = tester.mass_decode(fifty_index)
            for i in range(0, 50):
                temp = list_of_fifty_vector[i] * fifty_largest[:, i]
                product_vector[:, vocab_size * i:vocab_size * (i + 1)] = temp
            fifty_index = np.argsort(product_vector, kind = 'heapsort')[:, -50:]
            fifty_largest = np.sort(product_vector, kind = 'heapsort')[:, -50:]   
            index_previousw_matrix[j, :] = fifty_index / vocab_size
            index_currentw_matrix[j, :] = fifty_index % vocab_size
        # do a bottom-up search to figure out the french words index sequence for the largest probability
        final_translation_index_list.append(int(index_currentw_matrix[49, 49]))
        for i in range(48, -1, -1):
            index = index_previousw_matrix[i+1, i+1]
            previous_word_index = index_currentw_matrix[i, index]
            final_translation_index_list.append(int(previous_word_index))
        final_translation_index_list.reverse()
        print (final_translation_index_list)
        break
        # output is a french sentence which will be used 
    # once done, 


def array2indexedtuple(array): 
  return [val for val in enumerate(array[0])]

def get_bestN(array, N):
  indexed_array = array2indexedtuple(array)
  return sorted(indexed_array, key=lambda x: x[1])[:N]

# FIXME
# * how will we give input to decoder? text file? command line?

def decode():
    test()


    # for i in range(1000):
    #     source, target = train_feeder.get_batch(FLAGS.batch_size, en_length=en_length, fr_length=fr_length)
    #     print source
    #     print target
    # tester = SMT_Tester(en_length, hidden_dim, FLAGS.vocab_size, FLAGS.vocab_size, FLAGS.embedding_size)
    # output = []
    # for sentence in en_sentences:
    #     tester.encode(sentence)
    #     words = tester.decode()
    #     print words
        

        # terminated = False
        # while not terminated:
        #     for word in words:
        #         tester.decode()
        
    # prev = np.zeros([vocab_size,1])
    # for word in en_input:
    #     p = model.predict([word, prev])
    #     cur = softmax_sampling(p)
    #     prev = cur
    #     output.append(prev)
    # return output

if __name__ == "__main__":
    if FLAGS.plot_name is not None: 
        from keras.utils.visualize_util import plot

    if FLAGS.decode:
        decode()
    else:
        train()
