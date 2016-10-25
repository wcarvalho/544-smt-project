from data_feeder import DataFeeder
from seq_models import Seq2SeqModel

class Seq2SeqTrainer:

    def __init__(self):
        self.model = None
        self.data_feeder = None
        self.optimizer = None
        self.summary = None
        self.train_loss = []
        self.dev_loss = []

    def plot_loss(self):
        '''plot the training and validation losses to Tensorboard '''
        pass

    def decode_batch(self, source, target):
        pass

    def decode(self):
        pass

    def create_model(self):
        '''creates and initializes a Seq2SeqModel'''
        pass

    def create_optimizer(self, type='AdaGrad'):
        ''' creates and initializes an optimizer'''
        pass

    def create_data_feeder(self):
        ''' should return a DataFeeder'''
        pass

    def train(self):
        ''' the training loop '''
        self.create_model()
        self.create_optimizer()
        self.create_data_feeder()

        with tf.Session() as sess:
            global_step = 0
            while True:
                global_step += 1
                en_input, fr_input = self.data_feeder.get_batch()
                step_loss = self.optimizer.step(model=self.model,
                                    source=en_input,
                                    target=fr_input,
                                    session=sess)
                self.train_loss.append(step_loss)

                if global_step % 1000 == 0:
                    # s & t should befrom a feeder that serves validation set
                    err = self.decode_batch(source, target)
                    self.dev_loss.append(err)
                    self.plot_loss()

                if step_loss < threshold:
                    print('training finished')
                    break



def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()


''' tensorflow seems to have its own app management, the main(_) function
    will be called and all flags will be passed to it after app.run() '''
if __name__ == "__main__":
  tf.app.run()










