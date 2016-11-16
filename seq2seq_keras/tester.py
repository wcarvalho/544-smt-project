from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
import numpy as np

class SMT_Tester(object):
  """docstring for SMT_Tester"""
  def __init__(self, en_input_length, hidden_dim, en_vocab_size, fr_vocab_size, embedding_size=64):

    self.en_input_length = en_input_length
    self.hidden_dim = hidden_dim
    self.en_vocab_size = en_vocab_size
    self.fr_vocab_size = fr_vocab_size
    self.embedding_size = embedding_size

    self._build_encoder(en_input_length, hidden_dim, en_vocab_size, embedding_size)
    self._build_decoder(hidden_dim, embedding_size)
    self._build_fr_word_embedder(fr_vocab_size, embedding_size)
    
  def _build_encoder(self, input_length, hidden_dim, vocab_size, embedding_size=64):
    en = Input(shape=(input_length,), name='en_input_w')
    s = Embedding(vocab_size, embedding_size, input_length=input_length, name='en_embed_s')(en)
    h = LSTM(hidden_dim, return_sequences=False, name='hidden_h')(s)
    self.encoder = Model([encoder_input], [h])
    return self.encoder

  def _build_decoder(self, hidden_dim, embedding_size=64):
    decoder_input = Input(shape=(1, embedding_size+hidden_dim), name='decoder_input')
    z = LSTM(hidden_dim, return_sequences=False, name='hidden_z')(decoder_input)
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
    padding = np.zeros((1,self.embedding_size+self.hidden_dim-2*self.hidden_dim))
    return np.array([np.concatenate([self.recurrent_h,self.recurrent_h,padding],axis=-1)]).astype(np.float32)

  def _make_regular_batch(self, word_indx):
    indx_array = np.array([word_indx])
    embedded = self.fr_embedder.predict(indx_array)[0]
    return np.array([np.concatenate([self.recurrent_h,embedded],axis=-1)])

  def decode(self, word_indx = None, best_k=50):
    if word_indx is None: batch = self._make_initial_batch()
    else: batch = self._make_regular_batch(word_indx)
    probabilties = self.decoder.predict(batch)
    best = get_bestN(probabilties, best_k)
    return best

  def mass_decode(self, word_indices, best_k = 50)
    raise Exception("This has not yet been implemented!")
    for word in 

def array2indexedtuple(array): 
  return [val for val in enumerate(array[0])]

def get_bestN(array, N):
  indexed_array = array2indexedtuple(array)
  return sorted(indexed_array, key=lambda x: x[1])[:N]
