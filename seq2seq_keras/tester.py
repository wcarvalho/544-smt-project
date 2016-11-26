from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras import backend as K

import numpy as np
import sys
import copy
from beam_search import en2fr_beam_search, get_best, Node, Graph, get_worst, get_best_indices

class SMT(object):
  """docstring for SMT_Tester"""
  def __init__(self, en_input_length, hidden_dim, en_vocab_size, fr_vocab_size, embedding_size=64):


    self.en_input_length = en_input_length
    self.hidden_dim = hidden_dim
    self.en_vocab_size = en_vocab_size
    self.fr_vocab_size = fr_vocab_size
    self.embedding_size = embedding_size


    self._build_encoder(en_input_length, hidden_dim, en_vocab_size, embedding_size)
    self._build_decoder(hidden_dim, fr_vocab_size, embedding_size)
    self._build_fr_word_embedder(fr_vocab_size, embedding_size)

  def load_weights(self, weights):
    if weights is None: return
    self.encoder.load_weights(weights, by_name=True)
    self.decoder.load_weights(weights, by_name=True)

  def _build_encoder(self, input_length, hidden_dim, vocab_size, embedding_size=64):
    en = Input(shape=(input_length,), name='en_input_w')
    s = Embedding(vocab_size, embedding_size, input_length=input_length, name='en_embed_s')(en)
    h = LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    self.encoder = Model([en], [h])
    return self.encoder

  def _build_decoder(self, hidden_dim, vocab_size, embedding_size=64):
    decoder_input = Input(batch_shape=(1, 1, embedding_size+hidden_dim), name='decoder_input')
    self.z = LSTM(hidden_dim, name='hidden_z', stateful=True)
    z_out = self.z(decoder_input)
    p = Dense(vocab_size, activation='softmax', name='prob')(z_out)
    self.decoder = Model([decoder_input], [p])
    return self.decoder

  def _build_fr_word_embedder(self, vocab_size, embedding_size=64):
    fr_one_hot = Input(shape=(1,), name='fr_one_hot')
    fr_embedded = Embedding(vocab_size, embedding_size, input_length=1, name='fr_embedded')(fr_one_hot)
    self.fr_embedder = Model([fr_one_hot], fr_embedded)
    return self.fr_embedder

  def set_recurrent_h(self, x): self.recurrent_h = x

  def encode(self, input_sentence):
    self.recurrent_h = self.encoder.predict(input_sentence)
    return self.recurrent_h

  def _make_initial_batch(self):
    if self.hidden_dim < self.embedding_size: padding_size = self.embedding_size-self.hidden_dim
    else: padding_size = self.embedding_size

    padding = np.zeros((1,padding_size))
    return np.array([np.concatenate([self.recurrent_h,padding],axis=-1)]).astype(np.float32)

  def _make_regular_batch(self, word_indx):
    indx_array = np.array([word_indx])
    embedded = self.fr_embedder.predict(indx_array)[0]
    return np.array([np.concatenate([self.recurrent_h,embedded],axis=-1)])

  def decode(self, word_indx = None):
    # if word_indx is None: batch = self._make_initial_batch()
    if word_indx is None: batch = self._make_regular_batch(1)
    else: batch = self._make_regular_batch(word_indx)
    probabilties = self.decoder.predict(batch)
    return copy.deepcopy(probabilties), copy.deepcopy(self.get_decoder_rnn_states())

  def get_decoder_rnn_weights(self): 
    return copy.deepcopy(self.decoder.layers[1].get_weights())

  def set_decoder_rnn_weights(self, weights):
    self.decoder.layers[1].set_weights(weights)

  def get_decoder_rnn_states(self):
    return [i.eval() for i in self.z.states]

  def set_decoder_rnn_states(self, states):
    for i in range(len(states)):
      self.z.states[i].assign(states[i]).eval()

  def reset_states(self):
    self.decoder.layers[1].reset_states()

  def mass_decode(self, indices, states):
    probabilties = []
    post_states = []

    for i, s1 in zip(indices, states):
      self.set_decoder_rnn_states(s1)
      p, s2 = self.decode(i)
      probabilties.append(p)
      post_states.append(s2)

    return probabilties, post_states

  def beam_search(self, en_sentence, feeder, beam_size, max_search=100, verbosity=0):
    return en2fr_beam_search(self, feeder, en_sentence, beam_size, self.fr_vocab_size, max_search, verbosity)

  def greedy_search(self, en_sentence, length, verbosity=0):
    self.reset_states()
    self.encode(en_sentence)
    probabilties, weights = self.decode()
    best_indices, best_probabilities = get_best(probabilties, 1)
    del probabilties
    word_indices = list(best_indices)
    for i in range(1, length):
      probabilties, weights = self.decode(best_indices)
      del weights
      best_indices, best_probabilities = get_best(probabilties, 1)
      del probabilties
      word_indices.append(best_indices[0])

    return [word_indices]


def not_implemented(name): return "'"+name+"' has not yet been implemented!"