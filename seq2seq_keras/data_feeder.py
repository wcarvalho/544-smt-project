''' This module contains a set of functions from Tensorflow tutorial that
    handles downloading and unzipping WMT dataset. A DataFeeder class is 
    also defined which handles data preprocessing, tokenization, shuffling
    as well as feeding a minibatch '''

import re
import argparse
import os.path
from tensorflow.python.platform import gfile
import numpy as np

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
        self.en_vocab_inv, self.fr_vocab_inv = {}, {}
        self.en_data, self.fr_data = [], []
        self.pos = 0
        self.rand_idx = None
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
            self.en_vocab_inv = self.invert_vocab(self.en_vocab)
        if not self.fr_vocab:
            self.fr_vocab, _ = self.read_vocabulary(self.fr_vocab_path)
            self.fr_vocab_inv = self.invert_vocab(self.fr_vocab)

        print("start reading data... ")
        if not self.en_data:
            self.en_data = self.read_data(self.en_ids_path, max_num_samples)
        if not self.fr_data:
            self.fr_data = self.read_data(self.fr_ids_path, max_num_samples)


    def invert_vocab(self, vocab):
        return {vocab[key]: key for key in vocab}


    def words2feats(self, words, language="en"):
        if language == "en":
            words = [w if w in self.en_vocab else _UNK for w in words]
            return [self.en_vocab[w] for w in words]
        elif language == "fr":
            words = [w if w in self.fr_vocab else _UNK for w in words]
            return [self.fr_vocab[w] for w in words]
        else:
            raise ValueError('language can only be "en" or "fr"')


    def feats2words(self, feats, language="en", skip_special_tokens=False):
        if language not in ["en","fr"]:
            raise ValueError('language can only be "en" or "fr"')
        words = []
        if language == "en":
            try:
                words = [self.en_vocab_inv[f] for f in feats]
            except Exception as e:
                print self.en_vocab_inv
                print type(f)
                print type(self.en_vocab_inv.keys()[0])
                raise e
        elif language == "fr":
            words = [self.fr_vocab_inv[f] for f in feats]
        if skip_special_tokens:
            words = [w for w in words if w not in [_PAD, _GO, _EOS]]
        return words


    def get_size(self):
        if not self.en_data:
            raise Exception("Data is not loaded!")
        return len(self.en_data)


    def produce(self, batch_size=64):
        while 1:
            en, fr = self.get_batch(batch_size)
            fr_target = [seq[1:]+[PAD_ID] for seq in fr]
            en, fr = np.asarray(en), np.asarray(fr)
            fr_target = np.asarray(fr_target)
            weights = np.zeros_like(fr_target)
            weights[fr_target != PAD_ID] = 1
            # for i, v in enumerate(fr_target):
                # if v != PAD_ID: weights[i] = 1
            fr_target = np.expand_dims(fr_target, -1)
            yield ([en, fr], fr_target, weights)


    def get_batch(self, batch_size=64, en_length=40, fr_length=50):
        en = []
        fr = []
        if self.pos + batch_size >= len(self.en_data) - 1:
            self.pos = 0
        if self.pos == 0:
            self.rand_idx = np.random.permutation(self.get_size())
        for i in range(batch_size):
            tmp_en = self.en_data[self.rand_idx[self.pos + i]]
            tmp_fr = self.fr_data[self.rand_idx[self.pos + i]]
            while(len(tmp_en) < en_length):
                tmp_en.append(0)
            tmp_en = list(reversed(tmp_en))
            tmp_fr = [GO_ID] + tmp_fr + [EOS_ID]
            while(len(tmp_fr) < fr_length):
                tmp_fr.append(0)
            tmp_en = tmp_en[:en_length]
            tmp_fr = tmp_fr[:fr_length]
            en.append(tmp_en)
            fr.append(tmp_fr)
        self.pos += batch_size
        #print len(en)
        return (en, fr)


    # def read_data(self, ids_path, max_num_samples=0, offset=0):
    def read_data(self, ids_path, max_num_samples=0):
        # read from self.data_path
        # self.data = ...
        output = []
        file = gfile.GFile(ids_path)
        counter = 0
        for line in file:
            # line = self.__tokenize(line)
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
    df = DataFeeder(args.data_dir, args.prefix, args.vocab_size, 10000)
    print("English data size %d" % len(df.en_data))
    print("French data size %d" % len(df.fr_data))

    ([en, fr], _) = next(df.produce())
    print("Requested batch size: %d" % 10)
    print("English batch size %d" % len(en))
    print("French batch size %d" % len(fr))

    print("\ntesting words2feats and feats2words on an random sentence...")
    feats = df.words2feats("What a beautiful day !".split())
    print(feats)
    words = df.feats2words(feats)
    print(words)

    print("\ntesting feats2words on English training data...")
    for seq in en:
        sentence = df.feats2words(seq, language="en", skip_special_tokens=True)
        # English sequences were reversed when processed, so we need to reverse
        # them back before printing.
        sentence.reverse()
        print(" ".join(sentence))

    print("\ntesting feats2words on French training data...")
    for seq in fr:
        sentence = df.feats2words(seq, language="fr", skip_special_tokens=True)
        print(" ".join(sentence))

    print("finished")





