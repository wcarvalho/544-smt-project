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

from keras.layers import RepeatVector, Input, merge
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import rmsprop
from sklearn.manifold import TSNE
import numpy as np
import pylab

import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

# FIXME: this is sloppy, whoever did this
from data_feeder import *

from tester import SMT, stacked_lstm
from callbacks import MyTensorBoard
from testing import test_translation

def create_model(vocab_size, en_length, fr_length, hidden_dim):
    en = Input(shape=(en_length,), name='en_input_w')
    s = Embedding(vocab_size, FLAGS.embedding_size, input_length=en_length, mask_zero=True, name='en_embed_s')(en)
    h = stacked_lstm(s, hidden_dim, False, False, "hidden_h", FLAGS.num_layers)
    c = RepeatVector(fr_length, name='repeated_hidden_c')(h)
    fr = Input(shape=(fr_length,), name='fr_input_y')
    fr_encode = Embedding(vocab_size, FLAGS.embedding_size, input_length=fr_length, mask_zero=False, name='fr_embed_s')(fr)
    decode_input = merge([fr_encode, c], mode='concat', name='y_cat_c')
    z = stacked_lstm(decode_input, hidden_dim, True, False, "hidden_z", FLAGS.num_layers)
    p = TimeDistributed(Dense(vocab_size, activation='softmax'), name='prob')(z)
    model = Model(input=[en, fr], output=p)
    if FLAGS.plot_name:
        plot(model, to_file=FLAGS.plot_name+'.png', show_shapes=True)
    return model


def train_auto(FLAGS):
    # Prepare WMT data
    train_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                              prefix="giga-fren.release2",
                              vocab_size=FLAGS.vocab_size,
                              max_num_samples=FLAGS.max_train_data_size,
                              offset=FLAGS.offset)

    en_length, fr_length, hidden_dim = FLAGS.en_length, FLAGS.fr_length, FLAGS.hidden_dim

    model_train = create_model(FLAGS.vocab_size, en_length, fr_length, hidden_dim)
    model_train.compile(optimizer=rmsprop(lr=0.0005),
                        loss='sparse_categorical_crossentropy',
                        sample_weight_mode="temporal")
    print(model_train.summary())

    tester = SMT(en_length, hidden_dim, FLAGS.vocab_size, FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.num_layers, FLAGS.batch_size)

    if FLAGS.weights:
        model_train.load_weights(FLAGS.weights)

    tb_callback = MyTensorBoard(smt=tester, log_dir='../logs', histogram_freq=0, write_graph=False, write_images=False, flags=FLAGS)

    model_train.fit_generator(train_feeder.produce(FLAGS.batch_size), samples_per_epoch=train_feeder.get_size(),
                              nb_epoch=FLAGS.epochs, callbacks=[tb_callback])


# FIXME
# * how will we give input to decoder? text file? command line?
def decode(FLAGS):
    
    tester = SMT(FLAGS.en_length,
                FLAGS.hidden_dim,
                FLAGS.vocab_size,
                FLAGS.vocab_size,
                FLAGS.embedding_size,
                FLAGS.num_layers,
                FLAGS.batch_size)

    tester.load_weights(FLAGS.weights)

    test_feeder = DataFeeder(data_dir=FLAGS.data_dir,
                             prefix="newstest2013",
                             vocab_size=FLAGS.vocab_size)

    _ = test_translation(tester, test_feeder, FLAGS, nbatches=10, search_method=1)


def plot_tsne(features, dict, num_words=1000, output_path="embedding.png"):
    pylab.figure(figsize=(50, 50), dpi=100)
    max_x = np.amax(features, axis=0)[0]
    max_y = np.amax(features, axis=0)[1]
    pylab.xlim((-max_x, max_x))
    pylab.ylim((-max_y, max_y))
    pylab.scatter(features[:, 0], features[:, 1])
    for idx in range(min(len(features), num_words)):
        word = dict[idx]
        ords = [ord(c) for c in word]
        if max(ords) > 128:
            continue
        x = features[idx, 0]
        y = features[idx, 1]
        pylab.annotate(word, (x, y))
    pylab.savefig(output_path)


def tsne(FLAGS):
    model = Sequential()
    model.add(Embedding(FLAGS.vocab_size,
                  FLAGS.embedding_size,
                  name='en_embed_s'))
    model.load_weights(FLAGS.weights, by_name=True)
    feeder = DataFeeder(data_dir=FLAGS.data_dir,
                             prefix="newstest2013",
                             vocab_size=FLAGS.vocab_size)
    embedding = model.predict(range(FLAGS.vocab_size)).squeeze()
    tsne = TSNE(n_components=2, random_state=0, verbose=1, early_exaggeration=10)
    print("running t-sne")
    proj = tsne.fit_transform(embedding)
    print("finished")
    print("start plotting t-sne")
    plot_tsne(features=proj, dict=feeder.en_vocab_inv, num_words=3000,
              output_path="../logs/embeding.png")
    print("finished")


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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training.")
    
    parser.add_argument("--beam_size", type=int, default=50, help="Batch size to use during training.")
    parser.add_argument("--max_beam_search", type=int, default=100, help="Max iterations in beam search.")

    parser.add_argument("--en_length", type=int, default=40, help="Padded EN length to use during training.")
    parser.add_argument("--fr_length", type=int, default=50, help="Padded FR length to use during training.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Batch size to use during training.")
    parser.add_argument("--num_layers", type=int, default=3, help="Batch size to use during training.")
    

    parser.add_argument("--embedding_size", type=int, default=1024, help="Size of word embedding")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--data_dir", default="wmt", help="Data directory")
    parser.add_argument("--train_dir", default="wmt", help="Training directory.")
    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--offset", type=int, default=0, help="skip part of dataset")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200, help="How many training steps to do per checkpoint.")
    parser.add_argument("--decode", action='store_true', default=False, help="Set for interactive decoding.")
    parser.add_argument("--plot_name", type=str, help="base name for plots")
    parser.add_argument("-w", "--weights", type=str, help="weights file to load weights")
    parser.add_argument("--validation_frequency", type=int, help="how often to do validation", default=1000)
    parser.add_argument("--save_frequency", type=int, help="how often to do validation", default=5000)
    parser.add_argument("-e", "--epochs", type=int, help="how often to do validation", default=5)
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    parser.add_argument("-t", "--tsne", type=int, default=0)

    FLAGS = parser.parse_args()
    print(FLAGS)

    with sess.as_default():
        if FLAGS.plot_name:
            from keras.utils.visualize_util import plot
        if FLAGS.decode:
            decode(FLAGS)
        elif FLAGS.tsne:
            tsne(FLAGS)
        else:
            train_auto(FLAGS)
