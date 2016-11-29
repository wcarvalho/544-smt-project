import numpy as np
import sys
from collections import deque
from heapq import heappop, heappush

from beam_graph import Node, Graph


def en2fr_beam_search(smt, feeder, en_sentence, beam_size, vocab_size, max_search=50, verbosity=0):
    raise Exception("beam search not yet implemented")
    if verbosity > 0: print("start beam search...")

    batch_size = en_sentence.shape[0]
    graphs = [Graph(max_size=beam_size) for _ in range(batch_size)]

    # encode sentence into continuous vector
    smt.encode(en_sentence)
    probabilities, weights = smt.copy_decode()

    # sort
    sorted_i = np.argsort(probabilities, axis=1, kind = 'heapsort')
    sorted_p = np.sort(probabilities, axis=1, kind = 'heapsort')
    
    # reverse
    rsorted_i = np.fliplr(sorted_i)
    rsorted_p = np.fliplr(sorted_p)

    # get best
    best_i = rsorted_i[:,:beam_size]
    best_p = rsorted_p[:,:beam_size]

    print best_i
    probabilities_list, post_weights = smt.mass_decode(best_i, weights)
    sys.exit(0)
    # add best to graph for each batch
    for graph, indices, probabilities in zip(graphs, best_i, best_p):
      for indx, probability in zip(indices, probabilities):
          word = feeder.feats2words([indx], "fr")[0]
          word_node = Node(indx, np.log(probability), weights, word)
          graph.add(word_node)


    if verbosity > 0:
      graphs[0].print_best()
    stop_flags = [False]*batch_size
    j = 1

    nstops = 0
    while j < max_search:
      if verbosity > 1: print "\niter=",j
      
      for i, graph in enumerate(graphs):
        # if stop_flags[i]: continue
        previous_indices, previous_word_indices, previous_probabilities, previous_weights = graph.get_best()
        # print "pin", previous_indices
        print "pwo", previous_word_indices, type(previous_word_indices)
        # print "ppr", previous_probabilities

        # No more options, every available indx was EOS

        # get probability vectors and weights after feeding each word to SMT
        post_probability_set, post_weights = smt.mass_decode(previous_word_indices, previous_weights)

        noptions = len(post_probability_set)
        if noptions == 0: 
          stop_flags[i] = True
          continue

        # calculate all probabilities and put them in concatonated list
        product_vector = np.zeros((1, noptions * vocab_size))
        for i in range(noptions):
            # FIXME I wonder if we should add previous_probabilities to each candidate word, after all
            # beam search is a sort of greedy algorithm, for each candidate we should not consider
            # the total probability of the path from root to current leaf
            temp = np.log(post_probability_set[i]) # + previous_probabilities[i]

            product_vector[:, vocab_size * i:vocab_size * (i + 1)] = temp

        # sort concatonated list and get ordered integers
        unnormalized_indices = np.argsort(product_vector[:,:], kind = 'heapsort')[:, -beam_size:][0]

        # get indices for each word, parent_indices, and the corresponding probabilities
        new_indices = unnormalized_indices % vocab_size
        parent_rows = unnormalized_indices / vocab_size

        # print "pr", np.sort(product_vector[:,:], kind = 'heapsort')[:, -10:][0]
        # print "un", unnormalized_indices
        # print "pa", parent_rows

        probabilities = [post_probability_set[row][0][indx] for row, indx in zip(parent_rows, new_indices)]
        parents = [previous_indices[i] for i in parent_rows]
        parent_probabilities = [previous_probabilities[i] for i in parent_rows]

        # print "pa", parents
        # print "ne", new_indices
        for i in range(noptions):
            probability = np.log(probabilities[i])
            # +parent_probabilities[i]
            indx = new_indices[i]
            weights = post_weights[i]
            parent = parents[i]
            word = feeder.feats2words([indx], "fr")[0]
            word_node = Node(indx, probability, weights, word)
            graph.add(word_node, parent)
        j += 1
      if verbosity > 1:
        graph[0].print_best()
      if nstops == batch_size: break

    sequences = nodes.get_sequences(verbosity)
    return sequences

def get_best_multiple(array, beam_size, N):

  for i in range(beam_size):
    temp = all_probabilities[i] * original_word_probabilities[i]
    product_vector[:, vocab_size * i:vocab_size * (i + 1)] = temp
  new_indices, new_probabilities = get_best(product_vector, beam_size)

def get_best_indices(array, N, vocab_size): 
  x = np.argsort(array, kind = 'heapsort')[:, -N:][0] % vocab_size
  return x % vocab_size, x / vocab_size

def get_best(array, N):
  indices = np.argsort(array, kind = 'heapsort')[:, -N:][0]
  values = np.sort(array, kind = 'heapsort')[:, -N:][0]
  return indices, values

def get_worst(array, N):
  indices = np.argsort(array, kind = 'heapsort')[:, :N][0]
  values = np.sort(array, kind = 'heapsort')[:, :N][0]
  return indices, values