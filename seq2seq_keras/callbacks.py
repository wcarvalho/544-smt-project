from keras.callbacks import *
import os
import numpy as np
from tester import SMT
from data_feeder import DataFeeder

class MyTensorBoard(TensorBoard):


    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, flags=None):
        if not flags:
            raise Exception("flags cannot be None!")
        super(MyTensorBoard, self).__init__(log_dir, histogram_freq, write_graph, write_images)
        self.FLAGS = flags
        self.test_feeder = DataFeeder(data_dir=self.FLAGS.data_dir,
                                 prefix="newstest2013",
                                 vocab_size=self.FLAGS.vocab_size)


    def test(self):

        saved_weights = "temp.ckpt"
        self.model.save(saved_weights, overwrite=True)

        vocab_size = self.FLAGS.vocab_size
        embedding_size = self.FLAGS.embedding_size

        en_length = self.FLAGS.en_length
        hidden_dim = self.FLAGS.hidden_dim
        beam_size = self.FLAGS.beam_size

        tester = SMT(en_length, hidden_dim, vocab_size, vocab_size, embedding_size)
        tester.load_weights(saved_weights)

        for i in range(5):
            print("\nstart beam search...")
            en_sentence, _ = self.test_feeder.get_batch(1, en_length=en_length)
            en_sentence = np.array(en_sentence)
            en_str = self.test_feeder.feats2words(en_sentence[0], language='en', skip_special_tokens=True)
            print(" ".join(en_str))
            fr_sentence = tester.beam_search(en_sentence, self.test_feeder, beam_size=beam_size, verbosity=1)
            for fr in fr_sentence:
                fr_str = self.test_feeder.feats2words(fr, language='fr', skip_special_tokens=True)
                print(" ".join(fr_str))

        try:
            os.remove(saved_weights)
        except OSError:
            pass

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % 100 == 0:
            import tensorflow as tf
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = logs.get(name).item()
                summary_value.tag = name
                self.writer.add_summary(summary, batch)
            self.writer.flush()

        if (batch + 1) % 5000 == 0:
            save_path = "model-%d.ckpt" % (batch + 1)
            self.model.save(os.path.join(self.log_dir, save_path))

        if (batch + 1) % 100 == 0:
            self.test()



# class MyModelCheckpoint(ModelCheckpoint):
#
#     def on_batch_end(self, batch, logs={}):
#         if (batch + 1) % 1000 == 0:
#             self.on_epoch_end(batch, logs)
