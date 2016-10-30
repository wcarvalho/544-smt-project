''' This module contains a set of functions from Tensorflow tutorial that
    handles downloading and unzipping WMT dataset. A DataFeeder class is 
    also defined which handles data preprocessing, tokenization, shuffling
    as well as feeding a minibatch '''

import sys
import re
import tensorflow as tf

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

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

    def __init__(self, vocabulary_path, data_path):
        ''' all internal states are stored here '''
        self.vocabulary = {}
        self.vocabulary_path = vocabulary_path
        self.data_path = data_path


    def __tokenize(self,sentence):
        ''' tokenize preprocessed dataset, if available '''
        words = []
        for token in sentence.strip().split():
	        words.extend(_WORD_SPLIT.split(token))
	   	return words


    def __create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size,
                          tokenizer=None, normalize_digits=True):
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
				


    def __read_vocabulary(self, vocabulary_path):
        ''' originally initialize_vocabulary '''
        # reads from a vocabulary file into a dictionary
        pass
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


    def get_batch(self, batch_size=64):
        ''' returns a batach of given size '''
        #for i in range(64):
        pass
			
      
    def create_vocabulary(self):
			self.__create_vocabulary(vocabulary_path, data_path, 40000,
                          tokenizer=None, normalize_digits=False)
		
    def read_vocabulary(self, vocabulary_path):
		res_vocab, res_rev_vocab = self.__read_vocabulary(vocabulary_path)
		#print res_vocab
		#print res_rev_vocab
        
        
        
if __name__ == "__main__":
	vocabulary_path = sys.argv[1]
	data_path = sys.argv[2]
	df = DataFeeder(vocabulary_path, data_path)
	#df.create_vocabulary()
	df.read_vocabulary(vocabulary_path)
