import os
import os.path
import VocabTrie from vocab_trie

class ModelTrainer(object):

	MULTIVARIATE = 1;
	MULTINOMIAL = 2;
	SMOOTHED = 3;

	def __init__(self, type, dir):
		self.stop_words = self.fill_stop_list();
		self.train_dir = dir
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
		dirs = [dir for dir in os.listdir(self.train_dir)]
		for dir in dirs:
			total = len([n for n in os.listdir(self.train_dir + "/" + dir)])
			grand_total += total
			fraction = Container()
			fraction.numerator = 1
			fraction.denominator = total
			group_map[dir] = fraction
		return group_map

	def train_multivariate(self):
		words = VocabTrie(self.group_map)

		

	def train_multinomial(self):
		pass

	def train_smoothed(self):
		pass

class Container(object):
	pass

if __name__ == "__main__":
	trainer = ModelTrainer(1, "C:/NaiveBayes/20news-bydate-train")