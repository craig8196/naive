#This Trie is designed for the Bernoulli model

class TrieNode(object):
	
	def __init__(self, character, exists, index):
		self.character = character
		self.children = {}
		self.exists = exists

		if self.exists:
			self.occurrences = 1
			self.last_index = index
		else:
			self.occurrences = 0
			self.last_index = -1

	def add_child(self, character, exists, index):
		child = TrieNode(character, exists, index)
		self.children[character] = child

	def increment(self, index, model_type):
		if not self.exists:
			self.exists = True
		if model_type == 1:
			if index != self.last_index:
				self.occurrences += 1 
				self.last_index = index
		else:
			self.occurrences += 1

class VocabTrie(object):

	def __init__(self, group, model_type):
		self.root_node = TrieNode('', False, 0)
		self.group = group
		self.total = 0
		self.model_type = model_type

	def add_word(self, word, doc_index):
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1:
					current_node.children[c].increment(doc_index, self.model_type)
			else:
				if index == len(word) - 1:
					current_node.add_child(c, True, doc_index)
				else:
					current_node.add_child(c, False, doc_index)
			current_node = current_node.children[c]

	def get_probability_word_given_group(self, word):
		retvalue = 1.0 / (self.total + 1)
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1 and current_node.children[c].exists:
					retvalue = ((current_node.children[c].occurrences + 1) * 1.0) / (self.total + 1)
					break
			else:
				#return -1 if the word does not exist in the vocabulary
				break
			current_node = current_node.children[c]
		if retvalue == 0:
			print "not good"
		return retvalue

	def word_in_trie(self, word):
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1 and current_node.children[c].exists:
					return True
			else:
				#return -1 if the word does not exist in the vocabulary
				return False
			current_node = current_node.children[c]

class Container(object):
	pass



