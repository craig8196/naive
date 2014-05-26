import os
import os.path
from vocab_trie import VocabTrie
from vocab_trie import Container

class ModelTrainer(object):

	MULTIVARIATE = 1;
	MULTINOMIAL = 2;
	SMOOTHED = 3;

	def __init__(self, type, dir):
		self.stop_words = self.fill_stop_list();
		self.train_dir = dir
		self.dirs = [dir for dir in os.listdir(self.train_dir)]
		# self.group_map = self.init_group_map();
		self.group_map = {}

		if type == ModelTrainer.MULTIVARIATE:
			self.train_multivariate()
		elif type == ModelTrainer.MULTINOMIAL:
			self.train_multinomial()
		elif type == ModelTrainer.SMOOTHED:
			self.train_smoothed()

	def fill_stop_list(self):
		stop_trie = VocabTrie("", ModelTrainer.MULTIVARIATE)
		f = open('stoplist.txt', 'r')
		for line in f:
			stop_trie.add_word(line.rstrip().lower(), 0)
		return stop_trie

	def init_group_map(self):
		# group_map = {}
		# for my_dir in self.dirs:
		# 	total = len([n for n in os.listdir(self.train_dir + "/" + my_dir)])
		# 	fraction = Container()
		# 	fraction.numerator = 1
		# 	fraction.denominator = total
		# 	fraction.last_index = -1
		# 	group_map[my_dir] = fraction
		# return group_map

		group_map = []
		for my_dir in self.dirs:
			total = len([n for n in os.listdir(self.train_dir + "/" + my_dir)])
			fraction = Container()
			fraction.numerator = 1
			fraction.denominator = total

			fraction.name = my_dir
			group_map.append(fraction)
		return group_map

	def train_multivariate(self):
		print "\nBegin Multivariate Training"
		self.vocabulary = []
		total_words = 0
		self.total_files = 0
		for my_dir in self.dirs:
			print my_dir
			mytrie = VocabTrie(my_dir, ModelTrainer.MULTIVARIATE)
			for index, f in enumerate(os.listdir(self.train_dir + "/" + my_dir)):
				data_file = open(self.train_dir + "/" + my_dir + "/" + f, 'r')
				for line in data_file:
					for word in line.split():
						total_words += 1
						tempword = ''.join(c for c in word if c.isalpha())
						tempword = tempword.lower()
						if len(tempword) > 0 and not self.stop_words.word_in_trie(tempword):
							word_stored = mytrie.word_in_trie(tempword)
							if not word_stored:
								for k, v in self.group_map.iteritems():
									word_stored = v.word_in_trie(tempword)
									if word_stored:
										break
							if not word_stored:
								self.vocabulary.append(tempword)
							mytrie.add_word(tempword.lower(), index)
				mytrie.total = index + 1
				self.total_files += 1
			self.group_map[mytrie.group] = mytrie

	def train_multinomial(self):
		print "\nBegin Multinomial Training"
		self.total_files = 0
		for my_dir in self.dirs:
			print my_dir
			mytrie = VocabTrie(my_dir, ModelTrainer.MULTINOMIAL)
			for index, f in enumerate(os.listdir(self.train_dir + "/" + my_dir)):
				self.total_files += 1
				data_file = open(self.train_dir + "/" + my_dir + "/" + f, 'r')
				for line in data_file:
					for word in line.split():
						tempword = ''.join(c for c in word if c.isalpha())
						tempword = tempword.lower()
						if len(tempword) > 0 and not self.stop_words.word_in_trie(tempword):
							mytrie.add_word(tempword, index)
						mytrie.total += 1
			self.group_map[mytrie.group] = mytrie

	def train_smoothed(self):
		pass

class ModelTester(object):
	def __init__(self, type, trainer, dir):
		self.test_dir = dir
		self.dirs = [dir for dir in os.listdir(self.test_dir)]
		self.group_map = trainer.group_map
		self.total_files = trainer.total_files
		self.correct_map = {}

		if type == ModelTrainer.MULTIVARIATE:
			self.vocabulary = trainer.vocabulary
			self.test_multivariate()
		elif type == ModelTrainer.MULTINOMIAL:
			self.test_multinomial()
		elif type == ModelTrainer.SMOOTHED:
			self.test_smoothed()

	def test_multivariate(self):
		print "\nBegin Multivariate Testing"
		for my_dir in self.dirs:
			print my_dir
			total_correct = 0
			total = 0
			for index, f in enumerate(os.listdir(self.test_dir + "/" + my_dir)):
				words_in_doc = []
				data_file = open(self.test_dir + "/" + my_dir + "/" + f, 'r')
				temp_trie = VocabTrie(my_dir, ModelTrainer.MULTIVARIATE)
				for line in data_file:
					for word in line.split():
						tempword = ''.join(c for c in word if c.isalpha())
						tempword = tempword.lower()
						if len(tempword) > 0:
							temp_trie.add_word(tempword, index)
							words_in_doc.append(tempword)
				total_correct += self.is_max_correct(my_dir, temp_trie, words_in_doc)
				total += 1
			self.correct_map[my_dir] = [total_correct, total]
		print "\nMultivariate Results [correct, total]"
		for k, v in self.correct_map.iteritems():
			print k
			print v

	def test_multinomial(self):
		print "\nBegin Multinomial Testing"
		for my_dir in self.dirs:
			print my_dir
			total_correct = 0
			total = 0
			for index, f in enumerate(os.listdir(self.test_dir + "/" + my_dir)):
				words_in_doc = []
				data_file = open(self.test_dir + "/" + my_dir + "/" + f, 'r')
				for line in data_file:
					for word in line.split():
						tempword = ''.join(c for c in word if c.isalpha())
						tempword = tempword.lower()
						if len(tempword) > 0:
							words_in_doc.append(tempword)
				total_correct += self.is_max_correct_multinomial(my_dir, words_in_doc)
				total += 1
			self.correct_map[my_dir] = [total_correct, total]
		print "\nMultinomial Results [correct, total]"
		for k, v in self.correct_map.iteritems():
			print k
			print v

	def test_smoothed():
		pass

	def is_max_correct(self, group, temp_trie, words_in_doc):
		highest = 0
		max_group = ""
		# in_doc = []
		# not_in_doc = []
		# for word in self.vocabulary:
		# 	if temp_trie.word_in_trie(word):
		# 		in_doc.append(word)
		# 	else:
		# 		not_in_doc.append(word)
		for k, v in self.group_map.iteritems():
			probability = 1.0
			for word in words_in_doc:
				probability *= (1 + v.get_probability_word_given_group(word))
			# for word in in_doc:
			# 	probability *= v.get_probability_word_given_group(word)
			# for word in not_in_doc:
			# 	probability *= (1 - v.get_probability_word_given_group(word))
			probability *= (1.0 * v.total) / self.total_files
			if probability == 0:
				print "probability 0?"
			if probability > highest:
				highest = probability
				max_group = k
		if max_group == group:
			return 1
		else:
			return 0

	def is_max_correct_multinomial(self, group, words_in_doc):
		highest = 0
		max_group = ""
		for k, v in self.group_map.iteritems():
			probability = 1.0
			for word in words_in_doc:
				probability *= (1 + v.get_probability_word_given_group(word))
			probability *= (1.0 * v.total) / self.total_files
			if probability == 0:
				print "probability 0?"
			if probability > highest:
				highest = probability
				max_group = k
		if max_group == group:
			return 1
		else:
			return 0

if __name__ == "__main__":
	trainer = ModelTrainer(2, "C:/NaiveBayes/20news-bydate-train")
	tester = ModelTester(2, trainer, "C:/NaiveBayes/20news-bydate-test")
	trainer = ModelTrainer(1, "C:/NaiveBayes/20news-bydate-train")
	tester = ModelTester(1, trainer, "C:/NaiveBayes/20news-bydate-test")
	# print trainer.group_map["rec.sport.baseball"].get_probability_word_given_group("the")