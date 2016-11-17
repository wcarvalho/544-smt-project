from keras.callbacks import *
import os

class MyTensorBoard(TensorBoard):

    def on_batch_end(self, batch, logs={}):
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

        if (batch+1) % 50000 == 0:
            save_path = "model-%d.ckpt" % (batch+1)
            self.model.save(os.path.join(self.log_dir, save_path))
