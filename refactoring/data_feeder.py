''' This module contains a set of functions from Tensorflow tutorial that
    handles downloading and unzipping WMT dataset. A DataFeeder class is 
    also defined which handles data preprocessing, tokenization, shuffling
    as well as feeding a minibatch '''

import re
import argparse
import os.path
#import data_utils
from random import shuffle
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


class DataFeeder:

    def __init__(self, data_dir, prefix, vocab_size=10000, max_num_samples=1000000):
        ''' During initialization all file paths are created based on the root and prefix '''
        self.en_vocab, self.fr_vocab = {}, {}
        self.en_data, self.fr_data = [], []
        self.pos = 0
        self.vocab_size = vocab_size
        self.fr_raw_path = os.path.join(data_dir, prefix) + ".fr"
        self.en_raw_path = os.path.join(data_dir, prefix) + ".en"
        self.fr_vocab_path = os.path.join(data_dir, "vocab%d" % vocab_size) + ".fr"
        self.en_vocab_path = os.path.join(data_dir, "vocab%d" % vocab_size) + ".en"
        self.fr_ids_path = os.path.join(data_dir, prefix) + ".ids%d" % vocab_size + ".fr"
        self.en_ids_path = os.path.join(data_dir, prefix) + ".ids%d" % vocab_size + ".en"
        print("start preparing data...")
        self.prepare_data()
        # as a side effect of preparing data, self.data and self.vocabulary should've been
        # initialized. If they are empty, it means that all the temporary files exist already
        # and nothing has been done, so we need to read from the files manually.
        print("start reading vocabulary... ")
        if not self.en_vocab:
            self.en_vocab, _ = self.read_vocabulary(self.en_vocab_path)
        if not self.fr_vocab:
            self.fr_vocab, _ = self.read_vocabulary(self.fr_vocab_path)
        print("start reading data... ")
        if not self.en_data:
            self.en_data = self.read_data(self.en_ids_path, max_num_samples)
        if not self.fr_data:
            self.fr_data = self.read_data(self.fr_ids_path, max_num_samples)


    def get_batch(self, batch_size=64, en_length=40, fr_length=50):
        en = []
        fr = []
        sentence_index_list = [index for index in range(len(en_data))]
        for i in range(batch_size):
            if self.pos + i > len(self.en_data):
				shuffle(sentence_index_list)
                self.pos = 0
            tmp_en = self.en_data[sentence_index_list[self.pos + i]]
            tmp_fr = self.fr_data[sentence_index_list[self.pos + i]]
            while(len(tmp_en) < en_length):
                tmp_en.append(0)
            tmp_en = list(reversed(tmp_en))
            tmp_fr.insert(0, 0)
            while(len(tmp_fr) < fr_length):
                tmp_fr.append(0)
            tmp_en = tmp_en[:en_length]
            tmp_fr = tmp_fr[:fr_length]
            en.append(tmp_en)
            fr.append(tmp_fr)
        self.pos += batch_size
        #print len(en)
        return (en, fr)


    def read_data(self, ids_path, max_num_samples):
        # read from self.data_path
        # self.data = ...
        output = []
        file = gfile.GFile(ids_path)
        counter = 0
        for line in file:
            line = line.strip('\n').split(' ')
            line = [int(x) for x in line]
            output.append(line)
            counter += 1
            if counter % 10000 == 0:
                print ("%d lines read" % counter)
            if max_num_samples > 0 and counter >= max_num_samples:
                break
        return output


    def __tokenize(self,sentence):
        ''' tokenize preprocessed dataset, if available '''
        words = []
        for token in sentence.strip().split():
            words.extend(_WORD_SPLIT.split(token))
        return words


    def create_vocabulary(self, vocabulary_path, data_path, tokenizer=None, normalize_digits=True):
        '''create a vocabulary and save to disk'''
        # should call __tokenize() for each line of input
        # returns a {phrase : id} dictionary and also saves it to disk
        vocabulary = {}
        if not gfile.Exists(vocabulary_path):
            f = gfile.GFile(data_path, mode="rb")
            counter = 0
            for line in f:
                counter += 1
                if counter % 10000 == 0:
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
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
            vocab_list = _START_VOCAB + sorted(vocabulary, key = vocabulary.get, reverse = True)
            if len(vocab_list) > self.vocab_size:
                vocab_list = vocab_list[:self.vocab_size]
            vocab_file = gfile.GFile(vocabulary_path, mode= "wb")
            for w in vocab_list:
                vocab_file.write(w + b"\n")


    def read_vocabulary(self, vocabulary_path):
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


    def sentence_to_token_ids(self, sentence, vocabulary,
                              tokenizer=None, normalize_digits=True):
        ''' this function is taken directly from data_utils in the tensorflow tutorial'''
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.__tokenize(sentence)
        if not normalize_digits:
            return [vocabulary.get(w, UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

	def ids_to_sentence(self, ids, is_EN = True):
		if is_EN:
			return [en_vocab.keys()[en_vocab.values().index(indx)] for indx in ids]
		else:
			return [fr_vocab.keys()[fr_vocab.values().index(indx)] for indx in ids]

	def get_the_number_of_sentence(self, is_EN = True):
		if is_EN:
			return len(self.en_data)
		else:
			return len(self.fr_data)

    def data_to_token_ids(self, data_path, target_path, vocabulary_path,
                          tokenizer=None, normalize_digits=True):
        if not gfile.Exists(target_path):
            print("Tokenizing data in %s" % data_path)
            vocab, _ = self.read_vocabulary(vocabulary_path)
            with gfile.GFile(data_path, mode="rb") as data_file:
                with gfile.GFile(target_path, mode="w") as tokens_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 10000 == 0:
                            print("tokenizing line %d" % counter)
                        token_ids = self.sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


    def prepare_data(self, tokenizer=None):
        print("Create vocabularies of the appropriate sizes")
        if not gfile.Exists(self.fr_vocab_path):
            self.fr_vocab = self.create_vocabulary(self.fr_vocab_path, self.fr_raw_path, tokenizer)
        if not gfile.Exists(self.en_vocab_path):
            self.en_vocab = self.create_vocabulary(self.en_vocab_path, self.en_raw_path, tokenizer)

        print("Translate raw data to numbers using the vocabulary")
        if not gfile.Exists(self.fr_ids_path):
            self.data_to_token_ids(self.fr_raw_path, self.fr_ids_path, self.fr_vocab_path, tokenizer)
        if not gfile.Exists(self.en_ids_path):
            self.data_to_token_ids(self.en_raw_path, self.en_ids_path, self.en_vocab_path, tokenizer)
        print("finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A data feeder that iteratively yields a batch. The main function \
                                                  is for testing only')
    parser.add_argument('data_dir',
                        help="Path of a directory that contains the data set.")
    parser.add_argument('prefix',
                        help="Prefix of your raw data files. For example if you have two files wmt.en and \
                              wmt.fr the prefix should be wmt")
    parser.add_argument('vocab_size', type=int,
                        help="vocabulary size you'd like to process the data into.")
    args = parser.parse_args()
    print(args)

    # add your testing code here
    df = DataFeeder(args.data_dir, args.prefix, args.vocab_size, 10000000)
    print("English data size %d" % len(df.en_data))
    print("French data size %d" % len(df.fr_data))

    en, fr = df.get_batch(84)
    print("Requested batch size: %d" % 84)
    print("English batch size %d" % len(en))
    print("French batch size %d" % len(fr))





