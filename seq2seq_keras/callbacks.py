from keras.callbacks import *
import os
import numpy as np
from tester import SMT
from data_feeder import DataFeeder

class MyTensorBoard(TensorBoard):


    def __init__(self, smt, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, flags=None):
        if not flags:
            raise Exception("flags cannot be None!")
        super(MyTensorBoard, self).__init__(log_dir, histogram_freq, write_graph, write_images)

        self.smt = smt
        self.FLAGS = flags
        self.test_feeder = DataFeeder(data_dir=self.FLAGS.data_dir,
                                 prefix="newstest2013",
                                 vocab_size=self.FLAGS.vocab_size)


    def test(self):

        saved_weights = "temp.ckpt"
        self.model.save(saved_weights, overwrite=True)

        tester = self.smt
        tester.load_weights(saved_weights)

        for i in range(20):
            en_sentences, cr_fr_sentences = self.test_feeder.get_batch(self.FLAGS.batch_size, en_length=self.FLAGS.en_length)
            for i in en_sentences: i.reverse()
            en_sentences = np.array(en_sentences)
            cr_fr_sentences = np.array(cr_fr_sentences)
            fr_sentences = tester.greedy_search(en_sentences, self.FLAGS.fr_length, self.FLAGS.verbosity)
            for si in range(self.FLAGS.batch_size):
                en_sentence = en_sentences[si]
                fr_sentence = fr_sentences[si]
                cr_fr_sentence = cr_fr_sentences[si]

                en_sen = f2w(self.test_feeder, en_sentence)
                fr_sen = f2w(self.test_feeder, fr_sentence, lan="fr")
                cr_fr_sen = f2w(self.test_feeder, cr_fr_sentence, lan="fr")

                print("\nen: "+" ".join(en_sen))
                print("--\nfr len=%d: " % len(fr_sen) + " ".join(fr_sen))
                print("--\ncr fr len=%d: " % len(cr_fr_sen) + " ".join(cr_fr_sen))
            # fr_l = fr_length(cr_fr_sentence[0])


            # fr_sentence = tester.beam_search(en_sentence, self.test_feeder, beam_size=beam_size, max_search=fr_l, verbosity=self.FLAGS.verbosity)

            # for fr in fr_sentence:
            #     fr_str = self.test_feeder.feats2words(fr, language='fr', skip_special_tokens=True)
            #     print("fr len=%d: " % fr_length(fr_str) + " ".join(fr_str))

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

        if (batch + 1) % self.FLAGS.save_frequency == 0:
            save_path = "model-%d.ckpt" % (batch + 1)
            self.model.save(os.path.join(self.log_dir, save_path))

        if (batch + 1) % self.FLAGS.validation_frequency == 0:
            self.test()
            val_loss = self.model.evaluate_generator(generator=self.test_feeder.produce(self.FLAGS.batch_size),
                                          val_samples=self.test_feeder.get_size())
            print("|||Validataion Loss: %.3f" % val_loss)
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = val_loss
            summary_value.tag = "val_loss"
            self.writer.add_summary(summary, batch)
            self.writer.flush()

def f2w(feeder, sen, lan="en"): return feeder.feats2words(sen, language=lan, skip_special_tokens=True)

def fr_length(fr_indices):
    i = 0
    for indx in fr_indices:
        if indx == 0: break
        i=i+1

    return i


# class MyModelCheckpoint(ModelCheckpoint):
#
#     def on_batch_end(self, batch, logs={}):
#         if (batch + 1) % 1000 == 0:
#             self.on_epoch_end(batch, logs)
