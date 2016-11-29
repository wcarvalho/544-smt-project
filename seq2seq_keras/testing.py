import numpy as np

def test_translation(smt, feeder, FLAGS, nbatches=20, search_method=1, search_length=None):
  vocab_size = FLAGS.vocab_size
  embedding_size = FLAGS.embedding_size
  en_length = FLAGS.en_length
  hidden_dim = FLAGS.hidden_dim
  if search_length is None: search_length = FLAGS.fr_length 
 
  if search_method == 1: search_f=smt.greedy_search
  elif search_method == 2: search_f=smt.beam_search
  else: raise Exception("search method not supported!\nchoose 1 for greedy or 2 for beam search please")

  translations = []
  for i in range(nbatches):
    en_sentences, cr_fr_sentences = feeder.get_batch(FLAGS.batch_size, en_length=en_length)
    for i in en_sentences: i.reverse()
    en_sentences = np.asarray(en_sentences)
    cr_fr_sentences = np.asarray(cr_fr_sentences)
    if search_method == 1:
      fr_sentences = search_f(en_sentences, FLAGS.fr_length, FLAGS.verbosity)
    elif search_method == 2:
      fr_sentences = search_f(en_sentences, feeder, FLAGS.beam_size, search_length, FLAGS.verbosity)
    translations.append((en_sentences, fr_sentences))
    if FLAGS.verbosity > 0:
      for si in range(FLAGS.batch_size):
        en_sentence = en_sentences[si]
        fr_sentence = fr_sentences[si]
        cr_fr_sentence = cr_fr_sentences[si]

        en_sen = f2w(feeder, en_sentence)
        fr_sen = f2w(feeder, fr_sentence, lan="fr")
        cr_fr_sen = f2w(feeder, cr_fr_sentence, lan="fr")
        print("\n-------------")
        print("\nen: "+" ".join(en_sen))
        print("--\ncr fr len=%d: " % len(cr_fr_sen) + " ".join(cr_fr_sen))
        print("--\nfr len=%d: " % len(fr_sen) + " ".join(fr_sen))
  return translations


def f2w(feeder, sen, lan="en"): return feeder.feats2words(sen, language=lan, skip_special_tokens=True)

