import os
import os.path
from vocab_trie import VocabTrie

class ModelTrainer(object):

	MULTIVARIATE = 1;
	MULTINOMIAL = 2;
	SMOOTHED = 3;

	def __init__(self, type, dir):
		self.stop_words = self.fill_stop_list();
		self.train_dir = dir
		self.dirs = [dir for dir in os.listdir(self.train_dir)]
		self.group_map = self.init_group_map();

		if type == ModelTrainer.MULTIVARIATE:
			self.train_multivariate()
		elif type == ModelTrainer.MULTINOMIAL:
			self.train_multivariate()
		elif type == ModelTrainer.SMOOTHED:
			self.train_smoothed()

	def fill_stop_list(self):
		stop_list = []
		f = open('stoplist.txt', 'r')
		for line in f:
			stop_list.append(line.rstrip())
		return stop_list

	def init_group_map(self):
		group_map = {}
		for my_dir in self.dirs:
			total = len([n for n in os.listdir(self.train_dir + "/" + my_dir)])
			fraction = Container()
			fraction.numerator = 1
			fraction.denominator = total
			fraction.last_index = -1
			group_map[my_dir] = fraction
		return group_map

	def train_multivariate(self):
		self.words = VocabTrie(self.group_map)
		for my_dir in self.dirs:
			print my_dir
			for index, f in enumerate(os.listdir(self.train_dir + "/" + my_dir)):
				data_file = open(self.train_dir + "/" + my_dir + "/" + f, 'r')
				for line in data_file:
					for word in line.split():
						tempword = ''.join(c for c in word if c.isalpha())
						if len(tempword) > 0:
							self.words.add_word(tempword, index, my_dir)

	def train_multinomial(self):
		pass

	def train_smoothed(self):
		pass

class Container(object):
	pass

if __name__ == "__main__":
	trainer = ModelTrainer(1, "C:/NaiveBayes/20news-bydate-train")
	print trainer.words.get_probability_word_given_group("ball", "rec.sport.baseball")