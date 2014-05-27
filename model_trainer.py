from __future__ import division

import os
import re
import sys
import math
import os.path
from vocab_trie import VocabTrie
from vocab_trie import Container

DEBUG = True

def generate_word_set_from_file(file_path):
    result = set()
    with open(file_path, 'r') as f:
        for word in re.findall(r'[a-zA-Z]+', f.read()):
            result.add(word.lower())
    return result

def generate_word_list_from_text(text):
    result = []
    for word in re.findall(r'[a-zA-Z]+', text):
        result.append(word.lower())
    return result

class Classifier(object):
    """Create a Classifier object, add stopwords, train according to what
    will be used, initialize log space for faster computing, classify documents.
    """
    def __init__(self):
        self.total_docs = 0
        # tracks the total number of documents per category
        # used with both models
        self.cat_total_doc_counts = {}
        # list the total amount of word tokens in a category
        # used with multinomial model
        self.cat_total_word_counts = {}
        # list the number of times a word type occurs in the category and the number of documents that have the word
        # use with both models, word counts for multinomial model, and the other number for the bernoulli model
        self.cat_word_count_doc_count = {}
        # set of all stopwords
        self.stopwords = set()
    
    def add_stopwords(self, word_list):
        for word in word_list:
            self.stopwords.add(word)
    
    def train_all_from_directory(self, train_dir):
        """Pull in data from a directory of directories with documents.
        Both models are simultaneously trained.
        """
        category_dirs = os.listdir(train_dir)
        for category_dir in category_dirs:
            doc_names = os.listdir(train_dir + "/" + category_dir)
            for doc_name in doc_names:
                with open(train_dir + '/' + category_dir + '/' + doc_name, 'r') as f:
                    self.train(category_dir, generate_word_list_from_text(f.read()))
        if DEBUG:
            print "Total docs:", self.total_docs
            print "Total doc counts:", self.cat_total_doc_counts
            print "Total word counts:", self.cat_total_word_counts
            print "The rest:", self.cat_word_count_doc_count
    
    def train(self, category, words):
        new_words = []
        for w in words:
            if w not in self.stopwords:
                new_words.append(w)
        words = new_words
        # increment the number of documents a category has
        if category not in self.cat_total_doc_counts:
            self.cat_total_doc_counts[category] = 1
        else:
            self.cat_total_doc_counts[category] += 1
        self.total_docs += 1
        # increment total words
        if category not in self.cat_total_word_counts:
            self.cat_total_word_counts[category] = 0
        if category not in self.cat_word_count_doc_count:
            self.cat_word_count_doc_count[category] = {}
            
        word_dict = self.cat_word_count_doc_count[category]
        for word in words:
            # increment total word count
            self.cat_total_word_counts[category] += 1
            if word not in word_dict:
                word_dict[word] = [0, 0]
            # increment individual word count
            word_dict[word][0] += 1
        word_set = set(words)
        # increment word count of documents that contain that word
        for word in word_set:
            word_dict[word][1] += 1
        
    def initialize_log_space(self):
        # Calc. log(P(C)) used with both models
        self.log_p_c = {}
        for cat, cat_count in self.cat_total_doc_counts.items():
            self.log_p_c[cat] = math.log(cat_count) - math.log(self.total_docs)
        # Calc. log(P(F_i|C)) and log(1-P(F_i|C)), used with Multivariate Bernoulli where F is a feature (word type)
        self.log_f_given_c = {}
        self.log_notf_given_c = {}
        # Calc. log(P(W_i|C)), used with Multinomial Model where W is a word token
        self.log_w_given_c = {}
        for cat, words in self.cat_word_count_doc_count.items():
            self.log_f_given_c[cat] = {}
            self.log_notf_given_c[cat] = {}
            self.log_w_given_c[cat] = {}
            log_words = self.log_f_given_c[cat]
            log_not_words = self.log_notf_given_c[cat]
            log_w_given_c = self.log_w_given_c[cat]
            doc_total = self.cat_total_doc_counts[cat]
            word_total = self.cat_total_word_counts[cat]
            for word, counts in words.items():
                try:
                    log_w_given_c[word] = math.log(counts[0]) - math.log(word_total)
                    count = counts[1]
                    total = doc_total
                    if counts[1] == doc_total:
                        total += 1
                    if counts[1] == 0:
                        count = 1
                        total += 1
                    log_words[word] = math.log(count) - math.log(total)
                    log_not_words[word] = math.log(total - count) - math.log(total)
                except Exception as e:
                    print e
                    print counts[1], doc_total
        if DEBUG:
            print "P(C)", self.log_p_c
            print "P(F|C)", self.log_f_given_c
            print "P(W|C)", self.log_w_given_c
    
    def classify_multivariate_bernoulli(self, words):
        classification = ''
        largest = -sys.float_info.max
        for c in self.cat_total_doc_counts:
            calc = self.calc_multivariate_bernoulli(c, words)
            if calc > largest:
                largest = calc
                classification = c
        return classification
    
    def calc_multivariate_bernoulli(self, c, words):
        result = self.log_p_c[c]
        word_set = set(words)
        log_f_given_c = self.log_f_given_c[c]
        log_notf_given_c = self.log_notf_given_c[c]
        for w, num in word_set.items():
            if w in self.log_f_given_c:
                result += num
        return result
    
    def classify_multinomial(self, words):
        classification = ''
        largest = -sys.float_info.max
        for c in self.cat_total_doc_counts:
            calc = self.calc_multinomial(c, words)
            #~ print calc
            if calc > largest:
                #~ print largest, classification
                largest = calc
                classification = c
        return classification
    
    def calc_multinomial(self, c, words):
        result = self.log_p_c[c]
        for w in words:
            if w in self.log_w_given_c:
                result += self.log_w_given_c[w]
        return result
    
    def classify_multinomial2(self, words):
        print "Classifying:", words
        classification = ''
        largest = -sys.float_info.max
        for c in self.cat_total_doc_counts:
            calc = self.calc_multinomial2(c, words)
            print "Calc:", calc
            print "Class:", c
            if calc > largest:
                largest = calc
                classification = c
        return classification
    
    def calc_multinomial2(self, c, words):
        result = self.cat_total_doc_counts[c]
        my_words = self.cat_word_count_doc_count[c]
        count = 0
        for w in words:
            if w in my_words:
                result *= my_words[w][0]
                count += 1
        print count, result
        result = result/((self.cat_total_word_counts[c]**count)*self.total_docs)
        return result

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
        #   total = len([n for n in os.listdir(self.train_dir + "/" + my_dir)])
        #   fraction = Container()
        #   fraction.numerator = 1
        #   fraction.denominator = total
        #   fraction.last_index = -1
        #   group_map[my_dir] = fraction
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
        #   if temp_trie.word_in_trie(word):
        #       in_doc.append(word)
        #   else:
        #       not_in_doc.append(word)
        for k, v in self.group_map.iteritems():
            probability = 1.0
            for word in words_in_doc:
                probability *= (1 + v.get_probability_word_given_group(word))
            # for word in in_doc:
            #   probability *= v.get_probability_word_given_group(word)
            # for word in not_in_doc:
            #   probability *= (1 - v.get_probability_word_given_group(word))
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
    #~ trainer = ModelTrainer(2, "20news/train")
    #~ tester = ModelTester(2, trainer, "20news/test")
    #~ trainer = ModelTrainer(1, "20news/train")
    #~ tester = ModelTester(1, trainer, "20news/test")
    # print trainer.group_map["rec.sport.baseball"].get_probability_word_given_group("the")
    
    
    print "Created classifier."
    c = Classifier()
    print "Adding stopwords."
    #~ stopwords = generate_word_set_from_file('stopwords.txt')
    #~ c.add_stopwords(stopwords)
    print "Training classifier."
    c.train_all_from_directory('simple/train')
    print "Initializing log space."
    c.initialize_log_space()
    print "Testing."
    
    wins = 0
    losses = 0
    base_dir = 'simple/test'
    dirs = os.listdir(base_dir)
    print dirs
    for d in dirs:
        files = os.listdir(base_dir + '/' + d)
        for f in files:
            #~ f = files[i]
            with open(base_dir + '/' + d + '/' + f, 'r') as f2:
                cat = c.classify_multinomial(generate_word_list_from_text(f2.read()))
                if cat == d:
                    wins += 1
                else:
                    losses += 1
    print wins, losses
