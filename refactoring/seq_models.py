
''' This API has not been well planned out yet, it should be similar to
    the seq2seq_model defined in the TF tutorial, but data related methods
    such as get_batch() as been moved to a DataFeeder class, and optimizer
    is moved to a Optimizer class, so this class by itself only handles TF
    graph (neural structure) definition, initialization and resuming from
    checkpoints '''  


class Seq2SeqModel:

    def __init__(self):
        self.graph = None

    def create(self):
        pass

    def initialize_from_ckpt(self, path=""):
        pass
