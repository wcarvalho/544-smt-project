''' This module contains a set of functions from Tensorflow tutorial that
    handles downloading and unzipping WMT dataset. A DataFeeder class is 
    also defined which handles data preprocessing, tokenization, shuffling
    as well as feeding a minibatch '''


def maybe_download(directory, filename, url):
    pass


def gunzip_file(gz_path, new_path):
    pass


def get_wmt_enfr_train_set(directory):
    pass


def get_wmt_enfr_dev_set(directory):
    pass



class DataFeeder:

    def __init__(self):
        ''' all internal states are stored here '''
        self.vocabulary = {}
        self.vocabulary_path = ""
        self. data


    def __tokenize(self):
        ''' tokenize preprocessed dataset, if available '''
        pass


    def __create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size,
                          tokenizer=None, normalize_digits=True):
        '''create a vocabulary and save to disk'''
        # should call __tokenize() for each line of input
        # returns a {phrase : id} dictionary and also saves it to disk
        pass


    def __read_vocabulary(self, vocabulary_path):
        ''' originally initialize_vocabulary '''
        # reads from a vocabulary file into a dictionary
        pass


    def get_batch(self, batch_size=64):
        ''' returns a batach of given size '''





