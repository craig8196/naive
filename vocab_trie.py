#This Trie is designed for the Bernoulli model

class TrieNode(object):
	
	def __init__(self, character, exists, index):
		self.character = character
		self.children = {}
		self.exists = exists
		self.type_probability = 0
		self.token_probability = 0

		if self.exists:
			self.total_occurrences = 1
			self.doc_occurrences = 1
			self.last_index = index
		else:
			self.total_occurrences = 0
			self.doc_occurrences = 0
			self.last_index = -1

	def add_child(self, character, exists, index):
		child = TrieNode(character, exists, index)
		self.children[character] = child

	def increment(self, index):
		if not self.exists:
			self.exists = True
		if index != self.last_index:
			self.doc_occurrences += 1 
			self.last_index = index
		self.total_occurrences += 1

class VocabTrie(object):

	def __init__(self, group):
		self.root_node = TrieNode('', False, 0)
		self.group = group
		self.doc_total = 0
		self.word_total = 0

	def add_word(self, word, doc_index):
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1:
					current_node.children[c].increment(doc_index)
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

	def get_probability_type_given_group(self, word):
		retvalue = [1.0 , (self.doc_total + 1)]
		# retvalue = 0
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1 and current_node.children[c].exists:
					if current_node.children[c].type_probability == 0:
						current_node.children[c].type_probability = [((current_node.children[c].doc_occurrences + 1) * 1.0) , (self.doc_total + 1)]
					retvalue = current_node.children[c].type_probability
					break
			else:
				#return -1 if the word does not exist in the vocabulary
				break
			current_node = current_node.children[c]
		# if retvalue == 0:
		# 	print "not good"
		return retvalue

	def get_probability_token_given_group(self, word):
		retvalue = [1.0 , (self.word_total + 1)]
		# retvalue = 0
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1 and current_node.children[c].exists:
					if current_node.children[c].token_probability == 0:
						current_node.children[c].token_probability = [((current_node.children[c].total_occurrences + 1) * 1.0) , (self.word_total + 1)]
					retvalue = current_node.children[c].token_probability
					break
			else:
				#return -1 if the word does not exist in the vocabulary
				break
			current_node = current_node.children[c]
		# if retvalue == 0:
		# 	print "not good"
		return retvalue

	def get_probability_smoothed(self, word):
		retvalue = [0, self.word_total]
		current_node = self.root_node
		for index, c in enumerate(word):
			if c in current_node.children:
				if index == len(word) - 1 and current_node.children[c].exists:
					retvalue = [current_node.children[c].total_occurrences * 1.0 , self.word_total]
					break
			else:
				#return -1 if the word does not exist in the vocabulary
				break
			current_node = current_node.children[c]
		# if retvalue == 0:
		# 	print "not good"
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



