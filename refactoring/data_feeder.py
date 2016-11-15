''' This module contains a set of functions from Tensorflow tutorial that
    handles downloading and unzipping WMT dataset. A DataFeeder class is 
    also defined which handles data preprocessing, tokenization, shuffling
    as well as feeding a minibatch '''

import sys
import re
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def maybe_download(directory, filename, url):
    pass


def gunzip_file(gz_path, new_path):
    pass


def get_wmt_enfr_train_set(directory):
    pass


def get_wmt_enfr_dev_set(directory):
    pass



class DataFeeder:

    def __init__(self, train_path, dev_path, batch_size):
		''' all internal states are stored here '''
		self.vocabulary = {}
		self.data = []
		self.pos = 0
		self.batch_size = batch_size
		self.max_vocabulary_size = 40000
		self.train_path = train_path
		self.dev_path = dev_path
		self.fr_vocab_path = ''
		self.en_vocab_path = ''
		self.fr_train_ids_path = ''
		self.en_train_ids_path = ''
		self.fr_dev_ids_path = ''
		self.en_dev_ids_path = ''
        #self.buckets = buckets
        
        
    def get_batch(self):
		en = []
		fr = []
		for i in range(self.batch_size):
			if self.pos + i > len(self.data):
				self.pos = 0
			tmp = self.data[self.pos + i]
			tmp_en = tmp[0]
			tmp_fr = tmp[1]
			while(len(tmp_en) < 50):
				tmp_en.append(0)
			tmp_en = list(reversed(tmp_en))
			
			tmp_fr.insert(0, 0)
			while(len(tmp_fr) < 50):
				tmp_fr.append(0)
			en.append(tmp_en)
			fr.append(tmp_fr)
		self.pos += self.batch_size
		#print len(en)
		return [en, fr]
        
    def read_data(self):
		# read from self.data_path
		# self.data = ...
		index_dict = {}
		i = 0
		en_file = gfile.GFile(self.en_dev_ids_path)
		fr_file = gfile.GFile(self.fr_dev_ids_path)
		for line1 in en_file:
			line1 = line1.strip('\n').split(' ')
			line1 = [int(x) for x in line1]
			tmp = [line1]
			index_dict[i] = tmp
			i += 1
		for j in range(i):
			line2 = fr_file.readline()
			line2 = line2.strip('\n').split()
			line2 = [int(x) for x in line2]
			index_dict[j].append(line2)
		for index in range(len(index_dict)):
			self.data.append(index_dict[index])
		#print self.data
		return



    def __tokenize(self,sentence):
        ''' tokenize preprocessed dataset, if available '''
        words = []
        for token in sentence.strip().split():
	        words.extend(_WORD_SPLIT.split(token))
	   	return words


    def create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
        '''create a vocabulary and save to disk'''
        # should call __tokenize() for each line of input
        # returns a {phrase : id} dictionary and also saves it to disk
        if not gfile.Exists(vocabulary_path):
			f = gfile.GFile(data_path, mode="rb")
			counter = 0
			for line in f:
				counter += 1
				if counter % 100000 == 0:
					print ("Already processed %d lines" % counter)
				if tokenizer:
					tokens = tokenizer(line)
				else:
					tokens = self.__tokenize(line)
				for w in tokens:
					if normalize_digits:
						word = _DIGIT_RE.sub(b"0", w)
					else:
						word = w
					if word in self.vocabulary:
						self.vocabulary[word] += 1
					else:
						self.vocabulary[word] = 1
			vocab_list = _START_VOCAB + sorted(self.vocabulary, key = self.vocabulary.get, reverse = True)
			if len(vocab_list) > max_vocabulary_size:
				vocab_list = vocab_list[:max_vocabulary_size]
			vocab_file = gfile.GFile(vocabulary_path, mode= "wb")
			for w in vocab_list:
				vocab_file.write(w + b"\n")
				


    def initialize_vocabulary(self, vocabulary_path):
        ''' originally initialize_vocabulary '''
        # reads from a vocabulary file into a dictionary
        if not gfile.Exists(vocabulary_path):
			raise ValueError("Vocabulary file % not find. ", vocabulary_path)
        else:
			rev_vocab = []
			vocab = {}
			for line in gfile.GFile(vocabulary_path, mode = "rb"):
				line = line.strip()
				rev_vocab.append(line)
			for i in range(len(rev_vocab)):
				vocab[rev_vocab[i]] = i
			return vocab, rev_vocab


    def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
	  if not gfile.Exists(target_path):
		print("Tokenizing data in %s" % data_path)
		vocab, _ = self.initialize_vocabulary(vocabulary_path)
		with gfile.GFile(data_path, mode="rb") as data_file:
		  with gfile.GFile(target_path, mode="w") as tokens_file:
			counter = 0
			for line in data_file:
			  counter += 1
			  if counter % 100000 == 0:
				print("  tokenizing line %d" % counter)
			  token_ids = sentence_to_token_ids(line, vocab, tokenizer,
												normalize_digits)
			  tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
		
    def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
	  # Create vocabularies of the appropriate sizes.
	  self.fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
	  self.en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
	  create_vocabulary(self.fr_vocab_path, self.train_path + ".fr", fr_vocabulary_size, tokenizer)
	  create_vocabulary(self.en_vocab_path, self.train_path + ".en", en_vocabulary_size, tokenizer)

	  # Create token ids for the training data.
	  self.fr_train_ids_path = self.train_path + (".ids%d.fr" % fr_vocabulary_size)
	  self.en_train_ids_path = self.train_path + (".ids%d.en" % en_vocabulary_size)
	  data_to_token_ids(self.train_path + ".fr", self.fr_train_ids_path, self.fr_vocab_path, tokenizer)
	  data_to_token_ids(self.train_path + ".en", self.en_train_ids_path, self.en_vocab_path, tokenizer)

	  # Create token ids for the development data.
	  self.fr_dev_ids_path = self.dev_path + (".ids%d.fr" % fr_vocabulary_size)
	  self.en_dev_ids_path = self.dev_path + (".ids%d.en" % en_vocabulary_size)
	  data_to_token_ids(self.dev_path + ".fr", self.fr_dev_ids_path, fr_vocab_path, tokenizer)
	  data_to_token_ids(self.dev_path + ".en", self.en_dev_ids_path, en_vocab_path, tokenizer)

	  return (en_train_ids_path, fr_train_ids_path, en_dev_ids_path, fr_dev_ids_path, en_vocab_path, fr_vocab_path)
        
        
if __name__ == "__main__":
	train_path = sys.argv[1]
	dev_path = sys.argv[2]
	df = DataFeeder(train_path, dev_path, 64)
	df.read_data()
	#df.get_batch()
	#df.read_vocabulary()





