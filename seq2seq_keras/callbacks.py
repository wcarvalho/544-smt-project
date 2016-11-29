from keras.callbacks import *
import os
import numpy as np
from tester import SMT
from data_feeder import DataFeeder
from testing import test_translation

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
        self.smt.load_weights(saved_weights)

        translations = test_translation(self.smt, self.test_feeder, self.FLAGS, nbatches=20, search_method=1)


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
            print("||| Validataion Loss: %.3f" % val_loss)
            print("-------------------------------------------------------")
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = val_loss
            summary_value.tag = "val_loss"
            self.writer.add_summary(summary, batch)
            self.writer.flush()

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
