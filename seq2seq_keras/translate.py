# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from keras.models import Sequential
from keras.layers import RepeatVector, Input, merge
from keras.layers.core import Dense, MaxoutDense, Activation
from keras.engine.topology import Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from tensorflow import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.visualize_util import plot
from keras.models import Model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                                                    "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                                                    "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                                                        "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 10000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 10000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                                                        "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                                                        "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                                                        "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                                                        "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                                                        "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("quick_and_dirty", False,
                                                        "Quick & Dirty settings for fast testing")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


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
    print("printing model to an image")
    plot(model, to_file='model.png', show_shapes=True)
    return model



def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def one_hot(a, vocab_size):
    return (np.arange(vocab_size) == a[:,:,None]-1).astype(int)

def train():
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.

    print("Preparing WMT data in %s" % FLAGS.data_dir)
    tokenizer=None
    en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size, tokenizer, FLAGS.quick_and_dirty)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
                 % FLAGS.max_train_data_size)
    max_size = 0
    if FLAGS.quick_and_dirty: max_size = 1000
    dev_set = read_data(en_dev, fr_dev, max_size)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    print ("Finished reading data!")

    print("reading vocabulary")
    vocabulary_path_en = "./wmt/vocab10000.en"
    vocabulary_path_fr = "./wmt/vocab10000.fr"
    vocab_en, _ = data_utils.initialize_vocabulary(vocabulary_path_en)
    vocab_fr, _ = data_utils.initialize_vocabulary(vocabulary_path_fr)
    vocab_en = { vocab_en[key]:key for key in vocab_en}
    vocab_fr = {vocab_fr[key]:key for key in vocab_fr}
    print("finished")

    bucket_train_set = train_set[2]
    bucket_dev_set = dev_set[2]
    print(len(bucket_train_set))
    print(len(bucket_dev_set))
    print(len(bucket_train_set[0]))
    print(len(bucket_dev_set[0]))

    '''for feat in bucket_train_set:
        en, fr = feat
        en = " ".join([vocab_en[key] for key in en])
        fr = " ".join([vocab_fr[key] for key in fr])
        print("===============")
        print(en)
        print(fr)
        print(feat)

     '''
    hidden_dim = 1000
    en_length = 20
    fr_length = 25
    vocab_size = 10000

    source = [x[0] for x in bucket_train_set]
    target = [[1]+x[1] for x in bucket_train_set] 
    source = pad_sequences(source, maxlen=en_length, padding='pre')
    target = pad_sequences(target, maxlen=fr_length, padding='post')

    # embedding handles one hot coding, so source doesn't need it.
    target = one_hot(target, vocab_size)
    # target = target[..., np.newaxis]

    print(source.shape)
    print(target.shape)
    print(source[0])

    model = create_model(vocab_size, en_length, fr_length, hidden_dim)

    model.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy')

    print(model.summary())

    model.fit([source, target], target, nb_epoch=10, batch_size=32)






def self_test():
    """Test the translation model."""
    pass


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()
