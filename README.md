# 544-smt-project 

##Group #51 Neural Machine Translation

This is a Keras implementation of RNN Encoder-Decoder, written for as a final project for CSCI-544.

We built our model from basic Keras and Tensorflow building blocks, without using any pre-written sequence to sequence library. Our code is organized as follows:

1. `translate.py` is the entry point and takes many command line options.
2. `data_feeder.py` contains a DataFeeder class, a self sufficient class that handles processing, reading and producing batches from the data.
3. `callbacks.py` contains callback functions that are used in training sessions.
3. `beam_graph.py`, `beam_search.py`, `tester.py` and `testing.py` are related to the testing model and searching algorithms used in decoding.

Group members (in alphabetic order):

    Wilka Carvalho
    Yi Ren
    Shuyang Sheng
    Yaning Yu

For questions regarding the project, please leave comments or pull requests to:

    https://github.com/wcarvalho/544-smt-project