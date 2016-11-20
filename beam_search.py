import numpy as np

vocabulary_size = 10000
block_size = 50

def find_largest_fifty_pro(vector):
	fifty_index = np.argsort(vector, kind = 'heapsort')[:, -block_size:]
	fifty_largest = np.sort(vector, kind = 'heapsort')[:, -block_size:]
	return fifty_largest, fifty_index

def rnn(vector):
	a = []
	for i in range(0, block_size):
		a.append(np.random.rand(1, vocabulary_size))
	return a

def beam_search(source_vector):
	final_translation_wordno_list = []
	index_currentw_matrix = np.zeros((block_size, block_size)) 
	index_previousw_matrix = np.zeros((block_size, block_size))
	product_vector = np.zeros((1, block_size * vocabulary_size))
	fifty_largest, fifty_index = find_largest_fifty_pro(source_vector)
	index_currentw_matrix[0, :] = fifty_index

	for j in range(1, 50):
		list_of_fifty_vector = rnn(fifty_largest)
		for i in range(0, block_size):
			temp = list_of_fifty_vector[i] * fifty_largest[:, i]
			product_vector[:, vocabulary_size * i:vocabulary_size * ( i + 1 )] = temp
		fifty_largest, fifty_index = find_largest_fifty_pro(product_vector)
		index_previousw_matrix[j, :] = fifty_index / vocabulary_size
		index_currentw_matrix[j, :] = fifty_index % vocabulary_size

	final_translation_wordno_list.append(index_currentw_matrix[49, 49])
	for i in range(48, -1, -1):
		index = index_previousw_matrix[i+1, i+1]
		previous_wordno = index_currentw_matrix[i, index]
		final_translation_wordno_list.append(int(previous_wordno))
	final_translation_wordno_list.reverse()
	print final_translation_wordno_list

test = np.random.rand(1, vocabulary_size)
beam_search(test)