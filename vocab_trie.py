#This Trie is designed for the Bernoulli model

class TrieNode(object):
	
	def __init__(self, character, parent, exists, index, current_group, group_map):
		self.character = character
		self.children = {}
		self.group_map = {}

		if exists:
			self.group_map = group_map
			self.group_map[current_group].numerator += 1
			self.group_map[current_group].last_index = index 

	def add_child(self, character, exists, index, current_group, group_map):
		child = TrieNode(character, self, exists, index, current_group, group_map)
		self.children[character] = child

	def increment(self, index, current_group):
		if index != self.group_map[current_group].last_index:
			self.group_map[current_group].numerator += 1 
			self.group_map[current_group].last_index = index

class VocabTrie(object):

	def __init__(self, group_map):
		self.group_map = group_map
		self.root_node = TrieNode('', None, False, 0, "", self.group_map)

	def add_word(self, word, doc_index, current_group):
		current_node = self.root_node
		for index, c in enumerate(word):
			if current_node.children.keys.contains(c):
				if index == len(word) - 1:
					current_node.children[c].increment(doc_index, current_group)
			else:
				if index == len(word) - 1:
					current_node.add_child(c, True, doc_index, current_group, self.group_map)
				else:
					current_node.add_child(c, False, doc_index, current_group, self.group_map)
			current_node = current_node.children[c]

	def get_probability_word_given_group(self, word, group):
		current_node = self.root_node
		for index, c in enumerate(word):
			if current_node.children.keys.contains(c):
				if index == len(word) -1:
					return current_node.children[c].group_map[group].numerator / current_node.children[c].group_map[group].denominator
			else:
				#return -1 if the word does not exist in the vocabulary
				return -1;
			current_node = current_node.children[c]



