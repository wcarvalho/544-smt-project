import numpy as np
import sys
from collections import deque
from heapq import heappop, heappush

PI = 0 # probability index
II = 1 # word index
NI = 2 # node index

class Graph(object):
  """docstring for Graph"""

  def __init__(self, max_size=10):
    self.map = {}
    self.queue = deque([])
    self.max_size = max_size
    self.latest_indx = 0

  def size(self): return len(self.queue)
  def full(self): return self.size() >= self.max_size

  def add(self, node, parent=None):
    # increase global index for nodes
    self.latest_indx +=1
    node.set_indx(self.latest_indx)
    if parent is not None:
      parent = self.map[parent]
      node.set_parent(parent)

    removed = None
    if self.full():
      removed = self.queue.popleft()

    self.map[node.get_indx()] = node
    self.queue.append((node.get_probability(), node.get_word_indx(), node.get_indx()))

    # if removed EOS check that probability is lower than lowest in current queue. rebuild if not
    if removed is not None:
      i = removed[II]
      p = removed[PI]
      if i == 2:
        l = list(self.queue)
        sorted_queue = sorted(l, key=lambda x: x[PI])
        lowest_p_in_q = [0][PI]
        lowest_i_in_q = [0][II]
        # if p is greater than lowest, rebuild
        if p >= lowest_p_in_q:
          new_list = [i for i in self.queue if i[II] != lowest_i_in_q]
          new_list.append(removed)
          self.queue =  new_list


  def print_map(self):
    for i in self.map: print self.map[i]
  
  def print_best(self):
    for i in self.get_best_queue(): print self.map[i[NI]]

  # returns a sorted list of values
  def get_best_queue(self): 
    l = list(self.queue)
    sor = sorted(l, key=lambda x: x[PI], reverse=True)
    return sor[:self.max_size]
  
  def get_best(self): 
    best = self.get_best_queue()
    indices = [i[NI] for i in best if i[II] != 2]
    word_indices = [i[II] for i in best if i[II] != 2]
    probabilities = [i[PI] for i in best if i[II] != 2]
    weights = [self.map[i[NI]].get_weights() for i in best if i[II] != 2]
    return indices, word_indices, probabilities, weights

  def get_sequences(self):
    sequences = []
    best = self.get_best_queue()
    for i in best:
      sequence = self.get_sequence(i[NI])
      converted_sequence = [self.map[i].get_word_indx() for i in sequence]
      sequences.append(converted_sequence)
    return sequences

  def get_sequence(self, indx):
    node = self.map[indx]
    parent = node.get_parent()
    if parent == None: return [indx]
    else: 
      parent_indx = parent.get_indx()
      # print "parent", parent
      return self.get_sequence(parent_indx) + [indx]

class Node(object):
  """docstring for Node"""

  def __init__(self, word_indx, word_probability, weights, word=""):
    self.word_indx = word_indx
    self.probability = word_probability
    self.weights = weights
    self.children = []
    self.word = word
    self.parent = None
    self.number = 0

  def add_child(self, child):
    self.children.append(child)

  def set_parent(self, parent): self.parent = parent
  def set_indx(self, i): self.number = i

  def get_indx(self): return self.number
  def get_word_indx(self): return self.word_indx
  def get_word(self): return self.word
  def get_weights(self): return self.weights
  def get_probability(self): return self.probability
  def get_parent(self): return self.parent
  def delete_weights(self): del self.weights

  def __str__(self): 
    if self.parent is not None:
      return "parent:"+str(self.parent.get_indx())+"\t|word:"+str(self.get_indx()) +" "+ self.word +"\t|probability:"+str(self.probability)
    else: return "word:"+str(self.get_indx()) +" "+ self.word +"\t|probability:"+str(self.probability)


def en2fr_beam_search(smt, feeder, en_sentence, beam_size, vocab_size, max_search=100, verbosity=0):

    # initialize matrixs and vectors
    index_currentw_matrix = np.zeros((beam_size, beam_size)) 
    index_previousw_matrix = np.zeros((beam_size, beam_size))
    product_vector = np.zeros((1, beam_size * vocab_size))

    # encode sentence into continuous vector
    smt.encode(en_sentence)
    all_probabilities, weights = smt.decode()

    best_indices, best_probabilities = get_best(all_probabilities, beam_size)

    nodes = Graph(max_size=beam_size)

    for indx, probability in zip(best_indices, best_probabilities):
        word = feeder.feats2words([indx], "fr")[0]
        word_node = Node(indx, np.log(probability), weights, word)

        nodes.add(word_node)

    if verbosity > 0:
      nodes.print_best()

    j = 1
    while j < max_search:

        previous_indices, previous_word_indices, previous_probabilities, previous_weights = nodes.get_best()

        # No more options, every available indx was EOS
        if len(previous_word_indices) == 0: break

        # get probability vectors and weights after feeding each word to SMT
        post_probability_set, post_weights = smt.mass_decode(previous_word_indices, previous_weights)

        # calculate all probabilities and put them in concatonated list
        for i in range(beam_size):
            temp = np.log(post_probability_set[i]) + previous_probabilities[i]
            product_vector[:, vocab_size * i:vocab_size * (i + 1)] = temp

        # sort concatonated list and get ordered integers
        unnormalized_indices = np.argsort(product_vector, kind = 'heapsort')[:, -beam_size:][0]

        # get indices for each word, parent_indices, and the corresponding probabilities
        new_indices = unnormalized_indices % vocab_size
        parent_rows = unnormalized_indices / vocab_size
        probabilities = [post_probability_set[row][0][indx] for row, indx in zip(parent_rows, new_indices)]

        parents = [previous_indices[i] for i in parent_rows]
        parent_probabilities = [previous_probabilities[i] for i in parent_rows]

        for i in range(beam_size):
            probability = np.log(probabilities[i])+parent_probabilities[i]
            indx = new_indices[i]
            weights = post_weights[i]
            parent = parents[i]
            word = feeder.feats2words([indx], "fr")[0]
            word_node = Node(indx, probability, weights, word)
            nodes.add(word_node, parent)
        j += 1
        if verbosity > 1:
          nodes.print_best()


    sequences = nodes.get_sequences()
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