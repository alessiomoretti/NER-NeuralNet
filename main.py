# setting logging level
import logging, os
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 

import argparse

from model import *
from macros import *

from pypedia import PyPedia

import tensorflow as tf

def NER_pre_train(NER):
    # PRE-TRAINING
    # evalita2007 + evalita2009
    NER.training(restore=True, evalita=(EVALITA2007_TRAIN, EVALITA2007_TEST))
    NER.training(restore=True, evalita=(EVALITA2009_TRAIN, EVALITA2009_TEST))

def NER_training(NER, n_fold=True):
    # EVALITA2016
    if n_fold:
        NER.nfold_training(N=N_FOLD)
    else:
        NER.training(restore=True)

if __name__ == "__main__":

    # initializing argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", help="If set the pretrain routine is erformed", action="store_true")
    parser.add_argument("--train_nfold", help="If set the train routine is performed with n-fold cross validation", action="store_true")
    parser.add_argument("--train", help="If set the train routine is performed", action="store_true")
    parser.add_argument("--interactive", help="If set the interactive session is performed (model as per macro TRAINED_MDL", action="store_true")
    args   = parser.parse_args()

    # model initialization
    NER = NERModel(Embeddings(EMBEDDING))
    NER.build()

    # training routine
    if args.pretrain:
        NER_pre_train(NER)
    
    trained = False
    if args.train_nfold:
        NER_training(NER, n_fold=True)
        trained = True

    if args.train and not trained:
        NER_training(NER, n_fold=False)
        trained = True

    # interactive session
    if args.interactive:
        dbp = PyPedia(api=DBPEDIA_ENTRYPOINT)

        print("INTERACTIVE SESSION - insert sentence to be labelled or type 'quit' to exit")
        sentence = input(">>> ")
        while sentence != "quit":
            
            # printing out labelled 
            res = NER.interactive(sentence, model_path=TRAINED_MDL)
            for r in res:
                print(r[0], " - ", r[1])
            
            print("\n") 

            # printing out dbpedia results
            dbpedia = dbp.prepare_and_request(res, hits=DBPEDIA_MAX_HITS)
            for entity in dbpedia:
                if len(dbpedia[entity]) > 0:
                    print(entity)
                    for hit in dbpedia[entity]:
                        print(" " * 10, hit[0], " -> ", hit[1])
                else:
                    print(entity, "NIL")

                print("")

            print("--------------\n\n")
            sentence = input("NER > ")
