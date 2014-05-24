class ModelTrainer(object):

	MULTIVARIATE = 1;
	MULTINOMIAL = 2;
	SMOOTHED = 3;

	def __init__(self, type):
		if type == ModelTrainer.MULTIVARIATE:
			self.train_multivariate()
		elif type == ModelTrainer.MULTINOMIAL:
			self.train_multivariate()
		elif type == ModelTrainer.SMOOTHED:
			self.train_smoothed()

	def train_multivariate(self):
		pass

	def train_multinomial(self):
		pass

	def train_smoothed(self):
		pass