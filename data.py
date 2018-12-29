"""
This file contains a set of utilities to handle raw datasets
"""

import random
import numpy as np

from macros import *

import codecs


class Dataset:
    def __init__(self, embeddings, evalita = None, twitter = False, n_fold = None):
        """
        This class can be used to access via a unique interface
        to the dataset and to retrieve batches of sentences and words
        :param embeddings: Embeddings
        :param evalita: (optional) tuple => (evalita_test_path, evalita_train_path)
        :param twitter: (optional) to enable twitter char '#' '@'
        :param n_fold:  (optional) to enable n_fold enter the desired number of folds
        """

        # if we want to include twitter char '#' '@'
        self.twitter_enabled = twitter

        # building n_fold training set
        if n_fold is not None:
            train, _    = self.build_training_set(0)
            train, test = self.build_nfold_training(train, n_fold)
            self.trainingset = train
            self.testingset  = test
        else:
            # building testing and training sets
            if evalita is None:
                train, test  = self.build_training_set(SPLIT)                  
            if evalita:
                train  = self.read_evalita(evalita[0])
                test   = self.read_evalita(evalita[1])
            # a different split policy is applied when dataset is tweets only or evalita 
            self.trainingset    = train[int(len(train) * VALIDATION_SPLIT) : ]   # training set                                      
            self.validationset  = train[ : int(len(train) * VALIDATION_SPLIT)]   # validation set (from original training set)
            self.testingset     = test                                           # testing set

        self.embeddings  = embeddings

        # index of the last batch
        self.current     = 0
        # dictionary for the labels
        self.labels = LABELS_IOB
        # inverse dictionary for labels translation
        self.labels_translate = LABELS_BOI

        self.word_vector_len = len(self.embeddings.vocabulary["."])

        self.max_sequence = MAX_SEQ_LEN

    def shuffle_training_set(self):
        """
        This method can be used to shuffle the training set
        :return: None
        """
        # shuffling the training set
        random.shuffle(self.trainingset)
        random.shuffle(self.validationset)
        # resetting counter
        self.current = 0

    def get_next_batch(self, batch_size=1):
        """
        Return the next batch of word vectors and labels
        :param batch_size: int, the batch_size (if 0, the entire training set is provided)
        :return: list of tuple and list of lengths, [(word: str, vector: list, label: int)], [seq1_len ...]
        """
        batch_w   = []
        batch_l   = []
        lengths   = []

        # if the entire training set has been scanned
        if self.current >= len(self.trainingset):
            return (batch_w, batch_l), lengths
        # if the batch size is greater than the len... returning the entire trainingset
        if batch_size >= len(self.trainingset):
            batch_size = 0

        # retrieving sequences
        if batch_size == 0:
            sequences = self.trainingset
        else:
            sequences = self.trainingset[self.current : self.current + batch_size]

        # building batch
        (batch_w, batch_l), lengths = self.build_batch(sequences, 
                                                       self.embeddings.vocabulary,
                                                       self.labels, 
                                                       self.max_sequence, 
                                                       self.word_vector_len)

        # updating batch size
        self.current += batch_size
        
        return (batch_w, batch_l), lengths

    def get_test_batch(self):
        """
        Return the test batch of word vectors and labels
        :return: list of tuple and list of lengths, [(word: str, vector: list, label: int)], [seq1_len ...]
        """
        return self.build_batch(self.testingset, 
                                self.embeddings.vocabulary, 
                                self.labels, 
                                self.max_sequence, 
                                self.word_vector_len)

    def get_validation_batch(self):
        """
        Return the validation batch of word vectors and labels
        :return: list of tuple and list of lengths, [(word: str, vector: list, label: int)], [seq1_len ...]
        """
        return self.build_batch(self.validationset, 
                                self.embeddings.vocabulary, 
                                self.labels, 
                                self.max_sequence, 
                                self.word_vector_len)

    def build_nfold_training(self, training, N):
        """
        This utiity can be used to create an n-fold cross validation training
        and testing sets.
        :param training: list, sentences for the prepared training set
        :param N: integer, N fold parameter
        :return: tuple (training_sets, testing_sets)
        """

        # splitting the training set
        folds = [list(s) for s in np.array_split(training, N)]

        training = []
        testing  = []
        
        # populating training and testing sets
        for fold in folds:
            # test is the actual fold
            testing.append(fold)
            # preparing training subset
            tset = []
            for fset in folds:
                if fset != fold:
                    tset += fset
            # appending to training set
            training.append(tset)

        return training, testing
        

    def build_training_set(self, splitting_rate=0.2):
        """
        This utility can be used to retrieve the list of annotated words
        it also assume that hashtags and tags are entity, ignoring URLs in this
        implementation. 
        It return a tuple of list [tweet1 ...] for each tweet: ["w1 \t annotation"]
        
        :param splitting_rate: float, the splitting percentage
        :return: tuple, (training_set, testing_set)
        """
        t = self.get_tweets()
        n = self.get_gold_NER()

        # translation dictionary
        l = {"Organization" : "ORG",
             "Character"    : "PER",
             "Person"       : "PER",
             "Location"     : "LOC",
             "Event"        : "ENT",
             "Product"      : "PROD",
             "Thing"        : "ENT"}

        sequences = []

        for tid in t:
            if tid in n:
                tweet = t[tid]
                sequence = []

                start = 0
                for label in n[tid]:
                    # for each label parsing the previous sequence
                    seq = tweet[start : label[0]]
                    sequence.extend(self.parse_sequence(seq, self.twitter_enabled))
                    # parsing the respective entity
                    entity = tweet[label[0] : label[1] +1]
                    entity = entity.strip(":)[](!_?/\'\",«».").split(" ")
                    for e in entity:
                        if entity.index(e) == 0:
                            sequence.append(e + "\t" + "B-" + l[label[2]])
                        else:
                            sequence.append(e + "\t" + "I-" + l[label[2]])

                    start = label[1] + 1

                # finally parsing the remaining tweet
                if start < len(tweet) - 1:
                    seq = tweet[start:]
                    sequence.extend(self.parse_sequence(seq, self.twitter_enabled))

                sequences.append(sequence)

        # shuffling 
        random.shuffle(sequences)
        # retrieving test and training sets
        if splitting_rate > 0:
            splitting    = int(len(sequences) * splitting_rate)
            testing_set  = sequences[:splitting]
            training_set = sequences[splitting:]
        else:
            testing_set  = sequences
            training_set = sequences

        return training_set, testing_set

    @staticmethod
    def build_batch(batch, vocabulary, labels, max_len, word_len):
        """
        Given a batch as a list of tweets with words and labels separated by a tab
        it returns the batch of list of words, batch of list of labels and lengths
        :param batch: list 
        :param vocabulary: dictionary of word vectors
        :param labels: dictionary of labels
        :param max_len: integer, max sequence len
        :param word_len: integer, max word len
        :return: list of tuple and list of lengths, [(word: str, vector: list, label: int)], [seq1_len ...]
        """
        batch_w   = []
        batch_l   = []
        lengths   = []

        for seq in batch:
            sequence_w = []
            sequence_l = []
            for word in seq:
                [word, label] = word.split("\t")
                
                if word in vocabulary:
                    # retrieving word vector - lowerizing word to assest on italian tweets embedding
                    word_vector = vocabulary[word]
                elif word.lower() in vocabulary:
                    word_vector = vocabulary[word.lower()]
                else:
                    # uniform random vector if not in vocabulary
                    word_vector = np.random.random_sample(word_len)
                
                # retrieving and setting label vector
                lab = labels[label]
                
                sequence_w.append(word_vector)
                sequence_l.append(lab)
            
            # padding sequence
            while len(sequence_w) < max_len:
                word_vector = np.zeros(dtype=np.float32, shape=[word_len])
                sequence_w.append(word_vector)
                lab = 0
                sequence_l.append(lab)

            # updating batch and lengths
            batch_w.append(sequence_w)
            batch_l.append(sequence_l)
            lengths.append(max_len)

        return (batch_w, batch_l), lengths

    @staticmethod
    def parse_sequence(seq, twitter_enabled = False):
        sequence = []
        for s in seq.strip().split(" "):
            s = s.replace(".", ",")
            s = s.replace("«", ",")
            s = s.replace("»", ",")
            for o in s.split(","):
                # polishing
                if len(o) > 1:
                    o = o.strip(":)[](!_?/\'\",«».")

                if not twitter_enabled:
                    o = o.strip("#@")

                if o == "":
                    continue
                # if it has an '@' or '#' we can assume it as an entity (tbc)
                elif o[0] == "@" or o[0] == "#":
                    # if len(o) > 1 and twitter_enabled:
                    #    sequence.append(o + "\t" + "B-ENT")
                    continue
                # ignoring URLs and links
                elif o.find("http://") != -1 or o.find("https://") != -1:
                    continue
                else:
                    sequence.append(o + "\t" + "O")

        return sequence

    @staticmethod
    def get_tweets():
        """
        This utility can be used to retrieve the dictionary of available
        tweets and their IDs
        :return: dictionary [id] => tweet_body
        """
        tweets = dict()
        with open(TWEETS_FILE) as file:
            for line in file.readlines():
                line_s = line.split(TWEETS_SEPARATOR)
                if len(line_s) > 1:
                    tweets[line_s[0]] = line_s[1][:-1]
        return tweets

    @staticmethod
    def get_gold_NER():
        """
        This utility can be used to retrieve the dictionary of annotated
        tweets and their IDs
        :return: dictionary [id] => [(annotation)...]
        """
        ners = dict()
        with open(GOLD_NER_FILE) as file:
            for line in file.readlines():
                line_s = line.split("\t")

                tid = line_s[0]
                if tid not in ners:
                    ners[tid] = []
                ners[tid].append((int(line_s[1]), int(line_s[2]), line_s[4][:-1]))

            for n in ners:
                ners[n].sort()

        return ners

    @staticmethod
    def read_evalita(evalita_path):
        """
        This utility can be used to retrieve the sequences of annotated sentences
        from the EVALITA files provided with the format <word> <...> <...> <annotation>\n
        :param evalita_path: string, evalita file path
        :return: list of sentences in the form of ["<w1>\t<l1>", ...]
        """
        # preparing the final list of sequences
        evalitaset = []

        # iterating over different sentences
        with codecs.open(evalita_path, "r", encoding="utf-8", errors='ignore') as evalita:
            # initializing sentence list
            sentence = []

            # iterating over lines
            for eval in evalita.readlines():
                if len(eval) > 1:
                    eval = eval.strip("\n").split(" ")

                    # replacing 'GPE' with '
                    label = eval[3]
                    if "GPE" in label:
                        label = label.replace("GPE", "LOC")
                    
                    sentence.append(eval[0] + "\t" + label)
                    
                if len(eval) == 1:
                    evalitaset.append(sentence)
                    sentence = []

        # iterating over sentences 
        return evalitaset