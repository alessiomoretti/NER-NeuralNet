"""
Here a comprehensive list of all the macros used by the main of the application
"""

import os

# Tensorboard enabler
TENSORBOARD = True

# utils for logging
LOG_DIR      = os.path.join(os.getcwd(), "log")                                         # log dir for tensorboard
WS_METADATA  = os.path.join(LOG_DIR, "ws_metadata.tsv")                                 # word space metadata for tensorboard
MODEL_CKPT   = os.path.join(LOG_DIR, "model.ckpt")                                      # model default path
LOG_METRICS  = os.path.join(os.getcwd(), "metrics.txt")                                 # metrics log file                      

# utils for trained model (if any)
TRAINED_MDL  = os.path.join(os.getcwd(), "trained/model.ckpt")                          # previously trained model default path

# word embeddings
EMBEDDING = "./dataset/ws_201301.txt"                                                   # word embeddings source file

# labels
LABELS     = ["PER", "ORG", "LOC", "ENT", "PROD", "O"]  
LABELS_NOO = ["PER", "ORG", "LOC", "ENT", "PROD"]   
PER        = "PER"
LOC        = "LOC"
ORG        = "ORG"
ENT        = "ENT"
PROD       = "PROD"
O          = "O"
LABELS_IOB = {
                "O"     :  0,
                "B-PER" :  1,
                "I-PER" :  2,
                "B-LOC" :  3,
                "I-LOC" :  4,
                "B-ORG" :  5,
                "I-ORG" :  6,
                "B-PROD":  7,
                "I-PROD":  8,
                "B-ENT" :  9,
                "I-ENT" : 10
             }
LABELS_BOI  = { item[1] : item[0] for item in LABELS_IOB.items()}

# datasets max len
MAX_SEQ_LEN  = 130                       # EVALITA MAX LEN = 122

# N fold cross validation
N_FOLD = 6

# datasets (tweets)
TWITTER_CHARS    = True
GOLD_NER_FILE    = "./dataset/goldnerfile.tsv"                                              # annotated development dataset (no text)
TWEETS_FILE      = "./dataset/tweets.tsv"                                                   # annotated tweets ids
TWEETS_SEPARATOR = " > "
SPLIT            = 0.2                    # 20%(training set) -> testing set

# datasets (EVALITA 2007 2009)
EVALITA2007_TRAIN   = "./dataset/EVALITA2007/evalita07train.iob2"
EVALITA2007_TEST    = "./dataset/EVALITA2007/evalita07test.iob2"
EVALITA2009_TRAIN   = "./dataset/EVALITA2009/evalita09train.iob2"
EVALITA2009_TEST    = "./dataset/EVALITA2009/evalita09train.iob2"

# datasets training and validation
BATCH_SIZE          = 30
EPOCH               = 100
PLATEUX_BREAK       = 10
VALIDATION_SPLIT    = 0.3                 # 30%(set) -> validation set
GREEDY_APPROACH     = False

# DROPOUT
KEEP_PROBABILITY    = 0.7                                                                   # dropout keep probability          

# DBPEDIA 
DBPEDIA_ENTRYPOINT  = "http://lookup.dbpedia.org/api/search/KeywordSearch"
DBPEDIA_MAX_HITS    = 5
