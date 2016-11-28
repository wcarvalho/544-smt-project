from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras import backend as K

import numpy as np
import sys
import copy
from beam_search import en2fr_beam_search, get_best, Node, Graph, get_worst, get_best_indices

def stacked_lstm(input, hidden_dim, return_sequences, stateful, name, N):
    if N == 1: return LSTM(hidden_dim, return_sequences=return_sequences, stateful=stateful, name=name+"1")(input)

    h = LSTM(hidden_dim, return_sequences=True, stateful=stateful, name=name+"1")(input)
    if N > 2:
        for i in range(1, N-1):
            h = LSTM(hidden_dim, return_sequences=True, stateful=stateful, name=name+str(i+1))(h)
    h = LSTM(hidden_dim, return_sequences=return_sequences, stateful=stateful, name=name+str(N))(h)

    return h

class SMT(object):
  """docstring for SMT_Tester"""
  def __init__(self, en_input_length, hidden_dim, en_vocab_size, fr_vocab_size, embedding_size=64, num_layers=1, batch_size=5):


    self.en_input_length = en_input_length
    self.hidden_dim = hidden_dim
    self.en_vocab_size = en_vocab_size
    self.fr_vocab_size = fr_vocab_size
    self.embedding_size = embedding_size
    self.num_layers = num_layers
    self.batch_size = batch_size


    self._build_encoder(en_input_length, hidden_dim, en_vocab_size, embedding_size, num_layers)
    self._build_decoder(hidden_dim, fr_vocab_size, embedding_size, num_layers, batch_size)
    self._build_fr_word_embedder(fr_vocab_size, embedding_size)

    # self.map = {layer.name: layer for layer in self.decoder.layers}


  def load_weights(self, weights):
    if weights is None: return
    self.encoder.load_weights(weights, by_name=True)
    self.decoder.load_weights(weights, by_name=True)

  def _build_encoder(self, input_length, hidden_dim, vocab_size, embedding_size=64, num_layers=1):
    en = Input(shape=(input_length,), name='en_input_w')
    s = Embedding(vocab_size, embedding_size, input_length=input_length, mask_zero=True, name='en_embed_s')(en)
    h = stacked_lstm(s, hidden_dim, False, False, "hidden_h", num_layers)
    # h = LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    self.encoder = Model([en], [h])
    return self.encoder

  def _build_decoder(self, hidden_dim, vocab_size, embedding_size=64, num_layers=1, batch_size=5):

    decoder_input = Input(batch_shape=(batch_size, 1, embedding_size+hidden_dim), name='decoder_input')
    self.z = stacked_lstm(decoder_input, hidden_dim, False, True, "hidden_z", num_layers)
    # LSTM(hidden_dim, name='hidden_z', stateful=True)
    # z_out = self.z(decoder_input)
    p = Dense(vocab_size, activation='softmax', name='prob')(self.z)
    self.decoder = Model([decoder_input], [p])
    return self.decoder

  def _build_fr_word_embedder(self, vocab_size, embedding_size=64):
    fr_one_hot = Input(shape=(1,), name='fr_one_hot')
    fr_embedded = Embedding(vocab_size, embedding_size, input_length=1, name='fr_embedded')(fr_one_hot)
    self.fr_embedder = Model([fr_one_hot], fr_embedded)
    return self.fr_embedder

  def set_recurrent_h(self, x): self.recurrent_h = x

  def encode(self, input_sentence):
    self.reset_states()
    self.recurrent_h = self.encoder.predict(input_sentence)
    return self.recurrent_h

  def _make_initial_batch(self):
    if self.hidden_dim < self.embedding_size: padding_size = self.embedding_size-self.hidden_dim
    else: padding_size = self.embedding_size

    padding = np.zeros((1,padding_size))
    return np.array([np.concatenate([self.recurrent_h,padding],axis=-1)]).astype(np.float32)

  def _make_regular_batch(self, word_indx):
    # indx_array = np.ones((self.batch_size))
    if word_indx.shape[0] == 1:
      indx_array = np.repeat(word_indx, self.recurrent_h.shape[0], axis=-1)
    elif word_indx.shape[0] != self.recurrent_h.shape[0]:
      raise Exception("word_indx.shape[0] != self.recurrent_h.shape[0] " + str(word_indx.shape[0]) + "!=" + str(self.recurrent_h.shape[0]))
    else:
      indx_array = word_indx

    embedded = self.fr_embedder.predict(indx_array)
    recurrent_h = np.expand_dims(self.recurrent_h, 1)

    batch = np.concatenate([recurrent_h,embedded],axis=-1)
    # print ("embedded.shape", embedded.shape)
    # print ("recurrent_h.shape", recurrent_h.shape)
    return batch.astype(np.float32)

  def decode(self, word_indx = None):
    # if word_indx is None: batch = self._make_initial_batch()
    if word_indx is None: batch = self._make_regular_batch(np.array([1]))
    else: batch = self._make_regular_batch(word_indx)
    # print ("batch.shape", batch.shape)
    batch = np.reshape(batch, (self.batch_size, 1, self.embedding_size+self.hidden_dim))
    probabilties = self.decoder.predict(batch)
    return probabilties, self.get_decoder_rnn_states()

  def copy_decode(self, word_indx = None):
    return copy.deepcopy(self.decode(word_indx))

  def get_decoder_rnn_states(self):
    map = {}
    for layer in self.decoder.layers:
      if "hidden_z" in layer.name:
        map[layer.name] = [i.eval() for i in layer.states]
    return map

  def set_decoder_rnn_states(self, states):
    for layer in self.decoder.layers:
      if "hidden_z" in layer.name:
        input = states[layer.name]
        for i in range(len(input)):
          layer.states[i].assign(input[i]).eval()
        


  def reset_states(self):
    # self.encoder.layers[2].reset_states()
    self.decoder.reset_states()

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
    self.encode(en_sentence)
    probabilties, weights = self.decode()
    best_indices = np.zeros((len(probabilties), length))
    best_indices[:,0] = np.argsort(probabilties, axis=1, kind = 'heapsort')[:, -1]

    for j in range(1, length):
      probabilties, weights = self.decode(best_indices[:,j-1])
      best_indices[:,j] = np.argsort(probabilties, axis=1, kind = 'heapsort')[:, -1]

    return best_indices

def get_best(array, N):
  indices = np.argsort(array, kind = 'heapsort')[-N:]
  values = np.sort(array, kind = 'heapsort')[-N:]
  return indices, values

def not_implemented(name): return "'"+name+"' has not yet been implemented!"