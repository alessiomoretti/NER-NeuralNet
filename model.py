import numpy as np
import tensorflow as tf

from macros import *
from metrics import SimpleMetrics
from embeddings import Embeddings
from data import Dataset

class NERModel:
    def __init__(self, embeddings, evalita = None, twitter = False, debug=False):
        """
        This class define the neural network model as well
        as its operations for training, prediction and evaluation
        :param embeddings: Embeddings
        :param evalita: (optional) tuple => (evalita_test_path, evalita_train_path)
        :param twitter: (optional) to enable twitter char '#' '@' in dataset
        :param debug: (optional) to enable debug info and logs 
        """
        # initialization parameters
        self.hidden_size = embeddings.embed_dim
        self._embeddings = embeddings
        self.dataset     = Dataset(embeddings, evalita, twitter)
        self.ntags       = len(self.dataset.labels)

        # fixed parameters
        # TODO it is necessary to find a right forget_bias and learning_rate
        self.forget_bias = 1.0

        # initializing future variables
        self.embeddings  = None
        self.labels      = None
        self.output      = None
        self.predictions = None
        self.logits      = None
        self.loss        = None
        self.transition_params = None

        # dropout keep probability
        self.keep_prob   = None             

        # utils
        self.batch_size  = BATCH_SIZE
        self.labels_len  = self.dataset.max_sequence

        # training utilities
        self.optimizer    = None
        self.train_op     = None

        # initializing feed dictioanry
        self.feed_dict   = dict()

        # initializer
        self.init_g      = None
        self.init_l      = None

        # saver - initialization after variable init
        self.saver       = None

        # debug
        self.debug         = debug

        # tensorboard
        self.file_writer   = tf.summary.FileWriter(LOG_DIR)
        self.summary       = None

        # metrics
        self.metrics       = None

    def init_placeholders(self):
        """
        This method can be used as a routine to initialize the required placeholders
        :return: None
        """
        # self._sequence_lengths = tf.placeholder(tf.int32, shape=None)
        # self._word_embeddings  = tf.placeholder(tf.float32, shape=[None, None, self.hidden_size])
        # self._labels           = tf.placeholder(tf.int32, shape=[None, None])
        self.keep_prob           = tf.placeholder(tf.float32, shape=None)

    def set_feed_dictionary(self, embeddings = None, labels=None, keep_prob=None):
        """
        This method can be used to set the feed dictionary for each run of the
        session in order to get the correct parameters
        :param embeddings: tuple (word_embeddings, sequence_lengths)
        :param labels: list
        :param keep_prob: float, keep probability for dropout
        :return: None
        """
        if embeddings is not None:
            self.feed_dict[self.embeddings]  = embeddings[0]
            self.feed_dict[self.seq_lens] = embeddings[1]
            self.batch_size = len(embeddings[1])

        if labels is not None:
            self.feed_dict[self.labels] = labels
            self.labels_len = len(labels)

        if keep_prob is not None:
            self.feed_dict[self.keep_prob] = keep_prob

    def initialize_embeddings(self):
        """
        This method can be used to initialize the word embeddings  placeholder for each run
        :return: None
        """
        self.embeddings = tf.placeholder(name="embeddings", dtype=tf.float32, shape=[None, self.dataset.max_sequence, self.hidden_size])
        self.seq_lens = tf.placeholder(name="seq_lens", dtype=tf.int32, shape=[None])

    def initialize_labels(self):
        """
        This method can be used to initialize the labels for each run
        :return: None
        """
        self.labels = tf.placeholder(name="labels", dtype=tf.int32, shape=[None, self.dataset.max_sequence])

    def initialize_optimizer(self, gradient=False, start_learning_rate=0.001):
        """
        This method can be used to initialize the optimizer for the training step
        
        if GradientDescentOptimizer is chosen:
            according the official documentation for the exp decay of the learning rate
            decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            with a decay rate of 0.9 every 100 steps
        by default the AdamOptimizer is used - self exp decay of learning rate - 0.001

        :param gradient: (optional) boolean, if true a Gradient Descendent Optimizer is used according the learning rate
        :param start_learning_rate: (optional) float, starting learning rate - default is 0.1
        :return: None
        """

        # initializing global step
        self.global_step = tf.Variable(0, trainable=False)
        
        if gradient:
            learning_rate    = tf.train.exponential_decay(start_learning_rate, 
                                                          self.global_step, 
                                                          100,                    # decay every n steps
                                                          0.9,                    # decay rate
                                                          staircase=True)         # decay at discrete intervals

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer()

    def build_model(self):
        """
        This method is the main routine to build the model to train over the given dataset
        :return: None
        """
        
        self.embeddings = tf.reshape(self.embeddings, [-1, self.dataset.max_sequence, self.hidden_size])

        if self.debug:
            print("EMB ->", self.embeddings)

        # dropout on inputs
        self.embeddings = tf.nn.dropout(self.embeddings, keep_prob=self.keep_prob)

        # building the LSTM cells
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, self.forget_bias)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, self.forget_bias)

        # dropout on inputs - wrapper layer
        # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, keep_prob=self.keep_prob)
        # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, keep_prob=self.keep_prob)

        # building the bi-lstm rnn
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    self.embeddings,
                                                                    sequence_length=self.seq_lens,
                                                                    dtype=tf.float32)
        if self.debug:
            print("BI-LSTM build")
            print("FW ->", output_fw)
            print("BW ->", output_bw)

        # concatenating the outputs from the bw and fw cells
        self.output = tf.concat([output_fw, output_bw], -1)

        # reducing to the final scores with a final perceptron -> x * W + b = y
        self.W = tf.get_variable("W", shape=[2 * self.hidden_size, self.ntags],
                                 dtype=tf.float32)
        self.b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        # reshaping output in a flat manner (shape) => [seq_len, input_dimension]
        self.output = tf.reshape(self.output, [-1, 2*self.hidden_size])
        # retrieving predictions
        self.predictions = tf.matmul(self.output, self.W) + self.b
        # retrieving logits
        self.logits = tf.reshape(self.predictions, [-1, self.dataset.max_sequence, self.ntags])
        
        self.labels = tf.reshape(self.labels, shape=[-1, self.dataset.max_sequence])

        # using CRF (Conditional Random Field) to decode predictions
        # retrieving log likelihood and transition parameters (for future predictions)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                   self.labels,
                                                                                   self.seq_lens)
        if self.debug:
            print("CRF build")

        # getting loss
        self.loss = tf.reduce_mean(-log_likelihood)

        # optimization step
        self.train_op = self.optimizer.minimize(self.loss, self.global_step)

        # summary
        # TODO summary operation

    def initialize_placeholders(self):
        """
        Placeholders initialization routine
        """
        self.init_placeholders()
        self.initialize_embeddings()
        self.initialize_labels()

    def initialize_variables(self):
        """
        Variables initialization routine
        """
        self.init_g = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()

    def save_model(self, sess, path=MODEL_CKPT):
        """
        This utility can be used to save the model to the path specified
        :param path: the path to save the model to (by default is the one in macro MODEL_CKPT)
        :return: None
        """
        self.saver.save(sess, path)

    def restore_model(self, sess, path=MODEL_CKPT):
        """
        This utility can be used to restore the model from the path specified
        :param path: the path to restore the model from (by default is the one in macro MODEL_CKPT)
        :return: a positive value of restored 
        """
        try:
            self.saver.restore(sess, path)
            return True
        except:
            # no model to restore from
            return False


    def build(self):
        """
        Model build routine
        """
        self.initialize_placeholders()                    # initializing placeholders
        self.initialize_optimizer()                       # initializing the optimizer with exp learning rate decay
        self.build_model()                                # building the model
        self.initialize_variables()                       # global and local variables initializers
        self.saver = tf.train.Saver()
        print("[MODEL] build completed")


    def training_step(self, sess):
        """
        This method implements the base training step for the model
        :param sess: Session, the current session
        :return: float, training loss
        """

        # train operation, prediction, logits retrieval, transition_params init, training loss
        _, predictions, logits, transition_params, loss = sess.run([self.train_op,
                                                                    self.predictions,
                                                                    self.logits,
                                                                    self.transition_params,
                                                                    self.loss],
                                                                    feed_dict=self.feed_dict)
        # return loss
        return loss

    def predict_batch(self, sess, batch, len_vector, test=False):
        """
        This utility can be used to predict a batch of sentences 
        :param sess: Session, the current session
        :param batch: list, of sentences
        :param len_vector: list, of sentences lengths
        :param test: if True the batch is an already formatted batch
        :return: list, predictions - list of list of labels 
        """

        predictions = []
        sentences   = []
        lengths     = []

        if test:
            for sentence in batch:
                # formatting the sentence to be lowercase
                sentence = sentence.lower()
                # building word vector
                word_vector = []
                for word in sentence.split(" "):
                    try:
                        word_vector.append(self.dataset.embeddings.vocabulary[word])
                    except:
                        pass
                # padding to be max_sequence long
                while len(word_vector) < self.dataset.max_sequence:
                    word_vector.append(np.zeros(dtype=np.float32, shape=[self.hidden_size]))

                sentences.append(word_vector)
                lengths.append(len(word_vector))
        else:
            sentences = batch
            lengths   = len_vector

        # feeding dictionary
        self.set_feed_dictionary(embeddings=(sentences, lengths))

        # running session
        logits, transition_params = sess.run([self.logits,
                                              self.transition_params],
                                              feed_dict=self.feed_dict)
        # using CRF Viterbi decoding 
        viterbi_sequences = []
        for logit, seqlen in zip(logits, lengths):
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        # returning list of labelled sentences
        return viterbi_sequences

    def evaluate_model(self, sess, testing=None):
        """
        This utility allows the model to be evaluated using the dataset pre built 
        testing set. the F1_label_dict returned contains "PER", "ORG", "LOC", "PROD" and "ENT" labels
        :param sess: Session, the current session
        :param testing: (optional) list of tuple, the custom testing set
        :return: dictionary {"accuracy" : -, "precision" : -, "recall" : -, "F1" : -, "F1Labels" : - }
        """

        # retrieving test set
        if testing is None:
            (test_sentences, test_labels), test_lengths = self.dataset.get_test_batch()
        if testing is not None:
            (test_sentences, test_labels), test_lengths = self.dataset.build_batch(testing,
                                                                                   self._embeddings.vocabulary, 
                                                                                   self.dataset.labels, 
                                                                                   self.dataset.max_sequence,
                                                                                   self.dataset.word_vector_len)

        # keep probability 1.0
        self.set_feed_dictionary(keep_prob=1.0)

        # retrieving labelled sentences
        labels = []
        for i in range(0, int(len(test_sentences) / BATCH_SIZE) + 1):
            sentences = test_sentences[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            lengths   = test_lengths[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if len(sentences) > 0:
                labels += self.predict_batch(sess, sentences, lengths)
            else:
                break

        # running metrics
        self.metrics = SimpleMetrics(test_labels, labels)
        # retrieving accuracy
        accuracy, precision, recall, F1 = self.metrics.get_metrics(greedy=GREEDY_APPROACH)
        # retrieving F1 for labels PER, ORG and LOC
        F1_lab = dict()
        F1_lab["PER"]   = self.metrics.get_f1_label("PER", greedy=GREEDY_APPROACH)
        F1_lab["LOC"]   = self.metrics.get_f1_label("LOC", greedy=GREEDY_APPROACH)
        F1_lab["ORG"]   = self.metrics.get_f1_label("ORG", greedy=GREEDY_APPROACH)

        return {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "F1" : F1, "F1Labels" : F1_lab}

    def validate_model(self, sess):
        """
        This utility allows the model to be evaluated using the pre built validation set.
        :param sess: Session, the current session
        :return: float, the F1 measure upon which validation is performed
        """
        # retrieving validation set
        (test_sentences, test_labels), test_lengths = self.dataset.get_validation_batch()

        # keep probability 1.0
        self.set_feed_dictionary(keep_prob=1.0)

        # retrieving labelled sentences
        labels = []
        for i in range(0, int(len(test_sentences) / BATCH_SIZE) + 1):
            sentences = test_sentences[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            lengths   = test_lengths[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if len(sentences) > 0:
                labels += self.predict_batch(sess, sentences, lengths)
            else:
                break

        # running metrics
        self.metrics = SimpleMetrics(test_labels, labels)
        # retrieving accuracy
        _, _, _, F1 = self.metrics.get_metrics(greedy=GREEDY_APPROACH)
        return F1

    def training(self, restore=True, evalita=None):
        """
        This routine can be modified to run the training over the built model
        :param restore: (optional) boolean, if True the model is restored from a previous run
        :param evalita: (optional) tuple, (evalita_train,evalita_test) if a a certain evalita dataset wants to be instantiated
        :return: None
        """

        if evalita is not None:
            self.dataset = Dataset(self._embeddings, evalita=evalita, twitter=TWITTER_CHARS)

        best_model_metrics = {
            "accuracy" : 0.0,
            "precision": 0.0,
            "recall"   : 0.0,
            "F1"       : 0.0,
            "F1Labels" : None
        }

        best_score = 0.0

        # DEBUG 
        # with tf.Session() as sess:
        #     self.restore_model(sess)
        #     print(self.validate_model(sess))
        #     print(self.evaluate_model(sess))
        #     return

        # early stopping plateaux detection
        epoch     = 0
        plateaux  = 0
        # iterating over epochs
        for epoch in range(0, EPOCH):

            # shuffling training set
            self.dataset.shuffle_training_set()

            # initializing a new session
            with tf.Session() as sess:

                sess.run(self.init_g)               # global variables initializer
                sess.run(self.init_l)               # local  variables initializer
                print("[TRAIN] init completed")

                # restoring model
                if restore:
                    self.restore_model(sess)
                
                print("[TRAIN] running ...")
                run = 0

                # running training while examples to feed by the dataset 
                (sentences, labels), sequence_lengths = self.dataset.get_next_batch(batch_size=BATCH_SIZE)
                while len(sentences) != 0 and len(sentences) == BATCH_SIZE:
                    
                    # feeding dictionary
                    self.set_feed_dictionary(embeddings=(sentences, sequence_lengths),
                                             labels=labels,
                                             keep_prob=KEEP_PROBABILITY)
                    
                    # training step - retrieving loss
                    loss = self.training_step(sess)

                    # printing out metrics
                    print("[TRAIN] EPOCH {} - RUN {} - loss {}".format(epoch, run, loss))

                    # next step preparation
                    (sentences, labels), sequence_lengths = self.dataset.get_next_batch(batch_size=BATCH_SIZE)
                    run += 1

                # running validation over trained model
                F1 = self.validate_model(sess)
                print("[VALIDATE] F1 MEASURE: {}".format(F1))

                # comparison
                if (F1 > best_score) or (F1 == 0.0):
                    # resetting plateaux detection
                    plateaux = 0
                    # updating best score
                    best_score = F1
                    # saving model
                    self.save_model(sess)
                else:
                    # plateaux is detected
                    plateaux += 1
                    # early stopping due to no improvement
                    if plateaux == PLATEUX_BREAK:
                        break

        # running evaluation and storing metrics
        with tf.Session() as sess:
            self.restore_model(sess)
            best_model_metrics = self.evaluate_model(sess)
            sess.close()

        if LOG_METRICS is not None:
            # opening and writing to metrics file
            self.metrics.write_log_metric(best_model_metrics, LOG_METRICS, "EPOCH stopped {}".format(epoch))


    def nfold_training(self, N=N_FOLD, model_path=MODEL_CKPT):
        """
        Ad-hoc routine to run nfold cross validation over tweets dataset
        :param N: int, (optional) number of folds
        :param model_path: string, (optional) path to model 
        """
        # re instantiating dataset
        self.dataset = Dataset(self._embeddings, twitter=TWITTER_CHARS, n_fold=N)

        metrics = []
        nfold_counter = 0
        # running nfold
        for training, testing_batch in zip(self.dataset.trainingset, self.dataset.testingset):
            
            nfold_counter += 1
            print("---- N FOLD RUN ", nfold_counter, "----")

            # instantiating a new session
            with tf.Session() as sess:
                # preparing model
                sess.run(self.init_g)               
                sess.run(self.init_l)

                # restoring previous checkpoint      
                self.restore_model(sess, path=model_path)
                
                # batching
                for i in range(0, int(len(training) / BATCH_SIZE) + 1):
                    print("[TRAIN TWEETS] RUN ", i)
                    
                    # retrieving training batch
                    training_batch = training[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    # feeding dictionary
                    (sentences, labels), sequence_lengths = self.dataset.build_batch(training_batch,
                                                                                     self._embeddings.vocabulary, 
                                                                                     self.dataset.labels, 
                                                                                     self.dataset.max_sequence,
                                                                                     self.dataset.word_vector_len)
                    
                    self.set_feed_dictionary(embeddings=(sentences, sequence_lengths), labels=labels)
                    
                    # training step
                    self.training_step(sess)
                    
                # evaluating
                print("[EVALUATION TWEETS] running...")
                metrics.append(self.evaluate_model(sess, testing=testing_batch))


        if LOG_METRICS is not None:
            self.metrics.write_log_metric_nfold(metrics, N, LOG_METRICS)



    def interactive(self, sentence, model_path=MODEL_CKPT):
        """
        This routine can be used to interactively use the trained model to 
        label an input sentence.

        :param sentence: string, the sentence to be labelled
        :param model_path: (optional) string, the path from which restore the model
        :return: list tuples of labelled words [(w1, label1), ...]
        """

        # simple polishing
        # TODO improve polishing
        sentence = sentence.replace(",", "")
        sentence = sentence.replace(":", "")
        sentence = sentence.replace("'", " ")

        # sentence vector
        sentence_vector = sentence.split(" ")
        word_vector     = []

        # preparing word vector
        for word in sentence_vector:
            try:
                word_vector.append(self.dataset.embeddings.vocabulary[word.lower()])
            except:
                word_vector.append(np.random.random_sample(self.hidden_size))

        
        # normalizing word vector
        while len(word_vector) < self.dataset.max_sequence:
            word_vector.append(np.zeros(dtype=np.float32, shape=[self.hidden_size]))

        # running labelling session restoring model
        with tf.Session() as sess:
            # restoring routine
            restored = self.restore_model(sess, path=model_path)
            if not restored:
                print("No previous model to restore from!")
                return

            # setting feeding dictionary
            self.set_feed_dictionary(embeddings=([word_vector], [len(word_vector)]))
            # retrieving transition params
            logits, transition_params = sess.run([self.logits,
                                                  self.transition_params,],
                                                  feed_dict=self.feed_dict)
            viterbi_sequences = []
            for logit, seqlen in zip(logits, [len(word_vector)]):
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                viterbi_sequences += [viterbi_sequence]

            results = []
            for i in range(len(sentence_vector)):
                results.append((sentence_vector[i], self.dataset.labels_translate[viterbi_sequences[0][i]]))

            return results
