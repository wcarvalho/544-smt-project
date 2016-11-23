from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
import numpy as np
import sys
import copy


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
    z = LSTM(hidden_dim, name='hidden_z', stateful=True)(decoder_input)
    p = Dense(vocab_size, activation='softmax', name='prob')(z)
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
    if word_indx is None: batch = self._make_initial_batch()
    else: batch = self._make_regular_batch(word_indx)
    probabilties = self.decoder.predict(batch)
    return probabilties, self.get_decoder_rnn_weights()

  def get_decoder_rnn_weights(self): 
    return copy.deepcopy(self.decoder.layers[1].get_weights())

  def set_decoder_rnn_weights(self, weights):
    self.decoder.layers[1].set_weights(weights)

  def mass_decode(self, indices, weights):
    probabilties = []
    post_weights = []

    for i, w1 in zip(indices, weights):
      self.set_decoder_rnn_weights(w1)
      p, w2 = self.decode(i)
      probabilties.append(p)
      post_weights.append(w2)

    return probabilties, post_weights

  def beam_search(en_sentence, beam_size):

    return []

def not_implemented(name): return "'"+name+"' has not yet been implemented!"