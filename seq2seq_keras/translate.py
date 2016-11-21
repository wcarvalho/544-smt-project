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

import os
import sys
import numpy as np
from keras.layers import RepeatVector, Input, merge
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K

# FIXME: this is sloppy, whoever did this
from data_feeder import *

from tester import SMT_Tester
from callbacks import MyTensorBoard


def create_model(vocab_size, en_length, fr_length, hidden_dim):
    en = Input(shape=(en_length,), name='en_input_w')
    s = Embedding(vocab_size, FLAGS.embedding_size, input_length=en_length, mask_zero=True, name='en_embed_s')(en)
    h =LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    c = RepeatVector(fr_length, name='repeated_hidden_c')(h)
    fr = Input(shape=(fr_length,), name='fr_input_y')
    # fr_one_hot = Lambda(lambda x : K.one_hot(K.cast(x,'int32'), FLAGS.vocab_size), name="fr_input_y_one_hot")(fr)
    fr_encode = Embedding(vocab_size, FLAGS.embedding_size, input_length=fr_length, mask_zero=False, name='fr_embed_s')(fr)
    decode_input = merge([fr_encode, c], mode='concat', name='y_cat_c')
    z = LSTM(hidden_dim, return_sequences=True, name='hidden_z')(decode_input)
    p = TimeDistributed(Dense(vocab_size, activation='softmax'), name='prob')(z)
    model = Model(input=[en, fr], output=p)
    if FLAGS.plot_name:
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
    if FLAGS.plot_name:
        plot(model, to_file=FLAGS.plot_name+'_test.png', show_shapes=True)
    return model


def one_hot(x, vocab_size):
    ns, nt = x.shape
    output = K.zeros((ns, nt, vocab_size))


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = K.shape(labels_dense)[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[labels_dense.ravel()] = 1
    return labels_one_hot


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

    en_length, fr_length, hidden_dim = 40, 50, 256
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
            ## do_validation
            ## do_testing
            # optional: print("saving a model...")
            model_train.save(os.path.join(FLAGS.train_dir, "itr_%d.chkpoint" % (i+1)), overwrite=True)
    ## use best weights for testing

# def custom_loss(y_true, y_pred):
#     return K.sparse_categorical_crossentropy(y_pred, y_true, from_logits=True)


def train_auto(FLAGS):
    # Prepare WMT data
    train_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                              prefix="giga-fren.release2",
                              vocab_size=FLAGS.vocab_size,
                              max_num_samples=FLAGS.max_train_data_size)

    # test_feeder = DataFeeder(data_dir=FLAGS.data_dir,
    #                          prefix="newstest2013",
    #                          vocab_size=FLAGS.vocab_size)

    en_length, fr_length, hidden_dim = 20, 25, 256
    source, target = train_feeder.get_batch(FLAGS.max_train_data_size, en_length=en_length, fr_length=fr_length)
    source, target = np.asarray(source), np.asarray(target)
    target_output = np.expand_dims(target, -1)

    model_train = create_model(FLAGS.vocab_size, en_length, fr_length, hidden_dim)
    model_train.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy') # , metrics=['kullback_leibler_divergence']
    print(model_train.summary())

    # tensorboard callback
    tb_callback = MyTensorBoard(log_dir='../logs', histogram_freq=0, write_graph=False, write_images=False)
    # check point callback
    cp_callback = ModelCheckpoint(filepath="../logs/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                  monitor='val_loss',
                                  verbose = 0,
                                  mode="auto")
    model_train.fit([source, target], target_output, validation_split=0.1, nb_epoch=1, callbacks=[tb_callback, cp_callback])

def en2fr_beam_search(smt, en_sentence, beam_size, vocab_size):

    # initialize matrixs and vectors
    final_translation_index_list = []
    index_currentw_matrix = np.zeros((beam_size, beam_size)) 
    index_previousw_matrix = np.zeros((beam_size, beam_size))
    product_vector = np.zeros((1, beam_size * vocab_size))


    # encode sentence into continuous vector
    smt.encode(en_sentence)
    source_vector = smt.decode()
    
    # sort the source_vector and get the top beam_size largest probabilities
    fifty_index = np.argsort(source_vector, kind = 'heapsort')[:, -beam_size:]
    fifty_largest = np.sort(source_vector, kind = 'heapsort')[:, -beam_size:]

    # generate the matrix of top beam_size French words for each English word
    for j in range(1, beam_size):

        list_of_fifty_vector = smt.mass_decode(fifty_index)

        for i in range(0, beam_size):

            temp = list_of_fifty_vector[i] * fifty_largest[:, i]
            product_vector[:, vocab_size * i:vocab_size * (i + 1)] = temp
        
        fifty_index = np.argsort(product_vector, kind = 'heapsort')[:, -beam_size:]
        fifty_largest = np.sort(product_vector, kind = 'heapsort')[:, -beam_size:]

        index_previousw_matrix[j, :] = fifty_index / vocab_size
        index_currentw_matrix[j, :] = fifty_index % vocab_size
        fifty_index = index_currentw_matrix[j, :]


    # do a bottom-up search to figure out the french words index sequence for the largest probability
    print "index_previousw_matrix", index_previousw_matrix
    print "index_currentw_matrix", index_currentw_matrix
    final_translation_index_list.append(int(index_currentw_matrix[beam_size-1, beam_size-1]))
    for i in range(beam_size-2, -1, -1):
        index = int(index_previousw_matrix[i+1, i+1])
        previous_word_index = index_currentw_matrix[i, index]
        final_translation_index_list.append(int(previous_word_index))
    final_translation_index_list.reverse()
    return final_translation_index_list


def test(FLAGS):
    vocab_size = FLAGS.vocab_size
    embedding_size = FLAGS.embedding_size

    test_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                                         prefix="newstest2013",
                                         vocab_size=vocab_size)

    en_length = FLAGS.en_length
    hidden_dim = FLAGS.hidden_dim
    beam_size = FLAGS.beam_size
    saved_weights = FLAGS.weights

    tester = SMT_Tester(en_length, hidden_dim, vocab_size, vocab_size, embedding_size)
    tester.load_weights(saved_weights)

    for i in range(10):
        en_indices, _ = test_feeder.get_batch(1, en_length=en_length)

        en_indices = np.array(en_indices)
        fr_indices = en2fr_beam_search(tester, en_indices, beam_size, vocab_size)

        en_indices=en_indices[0]

        en_sent = indices2sent(en_indices, test_feeder.en_indx2vocab)
        fr_sent = indices2sent(fr_indices, test_feeder.fr_indx2vocab)
        print en_indices, en_sent
        print fr_indices, fr_sent
        break
    # print (final_translation_index_list)
        # output is a french sentence which will be used 
    # once done, 


# FIXME
# * how will we give input to decoder? text file? command line?
def decode(FLAGS):
    test(FLAGS)

def indices2sent(indices, indx2vocab):
    str = ""
    for i in indices:
        if i == 0: continue
        str += indx2vocab[i] + " "
    return str
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a basic RNN Encoder-Decoder')
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.99, help="Learning rate decays by this much.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use during training.")
    
    parser.add_argument("--beam_size", type=int, default=50, help="Batch size to use during training.")
    parser.add_argument("--en_length", type=int, default=40, help="Batch size to use during training.")
    parser.add_argument("--fr_length", type=int, default=50, help="Batch size to use during training.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Batch size to use during training.")
    

    parser.add_argument("--embedding_size", type=int, default=100, help="Size of word embedding")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--data_dir", default="wmt", help="Data directory")
    parser.add_argument("--train_dir", default="wmt", help="Training directory.")
    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200, help="How many training steps to do per checkpoint.")
    parser.add_argument("--decode", action='store_true', default=False, help="Set for interactive decoding.")
    parser.add_argument("--plot_name", type=str, help="base name for plots")
    parser.add_argument("--weights", type=str, help="weights file to load weights")
    FLAGS = parser.parse_args()
    print(FLAGS)

    if FLAGS.plot_name:
        from keras.utils.visualize_util import plot
    if FLAGS.decode:
        decode(FLAGS)
    else:
        train_auto(FLAGS)
