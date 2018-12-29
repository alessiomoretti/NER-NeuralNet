"""
In this module there is a simple implementation to run metrics and statistics
for a model evaluation. The module is written to be the more data-agnostic possible, 
it only requires:
    - batche of lists of correctly labelled data
    - batche of lists of output data
"""

import sys
import numpy as np

from macros import LABELS, LABELS_NOO, LABELS_IOB, LABELS_BOI

class SimpleMetrics:
    def __init__(self, labelled, outputs):
        """
        This class is built to provide a unique interface for the 
        metrics of the given model
        :param labelled: batch of correctly pre-labelled sequences
        :param outputs: outputs 
        """
        self.labelled_data = labelled
        self.output_data   = outputs

        self.accuracy      = None
        self.precision     = None
        self.recall        = None
        self.F1            = None

        # check for legally initialized parameters
        self.check_parameters()

        # initializing metrics dictionary
        self.metrics_strict = dict()
        self.metrics_greedy = dict()
        for m in {"TP", "FP", "FN", "TN"}:
            self.metrics_strict[m] = np.zeros(dtype=float, shape=[len(LABELS_NOO)])
            self.metrics_greedy[m] = np.zeros(dtype=float, shape=[len(LABELS_NOO)])

    def set_labelled(self, labelled):
        self.labelled_data = labelled

    def set_outputs(self, outputs):
        self.output_data = outputs

    def check_parameters(self):
        """
        This utility can be used to check for the parameters before running metrics 
        """
        exception = False

        if (len(self.labelled_data) == 0) or (len(self.output_data) == 0):
            raise Exception("Cannot run metrics over zero-len objects")
            exception = True

        if len(self.labelled_data) != len(self.output_data):
            raise Exception("Cannot run metrics on data of different lengths: labelled {} - model outputs {}".format(len(self.labelled_data), len(self.output_data)))
            exception = True

        for i in range(0, len(self.labelled_data)):
            if len(self.labelled_data[i]) != len(self.output_data[i]):
                raise Exception("Cannot run metrics on data of differente lengths: at index {} labelled {} - model outputs {}".format(i, len(self.labelled_data[i]), len(self.output_data[i])))
                exception = True
        
        if exception:
            # terminating program if an exception occured
            sys.exit(-1)

    def compute_metrics(self):
        """
        This method can be used to retrieve statistics according to strict and greedy policies
        :return: None
        """
        # check for legally initializied parameters
        self.check_parameters()
        
        # computing metrics for each output / labelled
        for labelled, output in zip(self.labelled_data, self.output_data):
            # we have to check for each correctly classified entity
            # Labelled: B-Per I-Per I-Per O O O B-Loc I-Loc
            # Output:   B-Per I-Per I-Per O O O B-Loc O
            #
            # Strict:   TP=1 FN=1 ... 
            # Greedy:   TP=2 FN=0 ...

            # 1. SPLITTING&COMPARISON according to oracle - for greedy vs strict TP / FN
            i = 0
            entities = [] 
            entity_l = []  # entity labelled by oracle
            entity_o = []  # entity mapped in system output

            while i < len(labelled):
                    
                # 1.1 SPLITTING according to oracle
                if labelled[i] != 0:                                    # outer token -> excluded
                    label = LABELS_BOI[labelled[i]]
                    outp  = LABELS_BOI[output[i]]
                    if "B" in label:                                    # begin token
                        if len(entity_l) != 0:                          # begin1. resetting entities (if any)
                            entities.append((entity_l, entity_o))       
                            entity_l = []
                            entity_o = []
                        entity_l = [label]                              # begin2. creating new entitites
                        entity_o = [outp]
                    elif "I" in label:                                  # inner token
                        entity_l.append(label)
                        entity_o.append(outp)
                    
                if labelled[i] == 0 or (i+1) == len(labelled):          # if outer token or end of sentence
                    if len(entity_l) != 0:                              # resetting entities (if any)
                        entities.append((entity_l, entity_o))
                        entity_l = []
                        entity_o = []

                i += 1

            # 1.2. COMPARISON according to oracle - for greedy vs strict TP and FN
            # iterating over entities
            for entity_oracle, entity_system in entities:

                # (retrieving label id from xxx in B-xxx)
                lab_id = LABELS.index(entity_oracle[0][2:])
                # if B-xxx is not exact match -> no overlap
                if entity_oracle[0] != entity_system[0]:
                    self.metrics_strict["FN"][lab_id] += 1
                    self.metrics_greedy["FN"][lab_id] += 1
                else:
                    # exact match
                    if entity_oracle == entity_system:
                        self.metrics_strict["TP"][lab_id] += 1
                        self.metrics_greedy["TP"][lab_id] += 1
                    # overlapping only
                    else:
                        self.metrics_strict["FN"][lab_id] += 1
                        self.metrics_greedy["TP"][lab_id] += 1

            # 2. UPDATING FP and FN 
            # iterating over output
            for lab, out in zip(labelled, output):
                if out == 0:
                    if out == lab:
                        tn  = np.array([1] * len(LABELS_NOO))
                        self.metrics_strict["TN"] += tn
                        self.metrics_greedy["TN"] += tn
                else:
                    if lab == 0:
                        fp = np.array([1] * len(LABELS_NOO))
                        self.metrics_strict["FP"] += fp
                        self.metrics_greedy["FP"] += fp

    def get_metrics(self, greedy=False):
        """
        This method can be used to return the metrics (Accuracy, Precision, Recall, F1)
        :param greedy: (optional) boolean, wheter or not return the greedy metrics
        :return: metrics as a tuple (A, P, R, F1)
        """

        # computing metrics
        self.compute_metrics()

        if greedy:
            # excluding 
            TP = np.sum(self.metrics_greedy["TP"], dtype=np.float) 
            TN = np.sum(self.metrics_greedy["TN"], dtype=np.float)
            FP = np.sum(self.metrics_greedy["FP"], dtype=np.float)
            FN = np.sum(self.metrics_greedy["FN"], dtype=np.float)
        else:
            TP = np.sum(self.metrics_strict["TP"], dtype=np.float) 
            TN = np.sum(self.metrics_strict["TN"], dtype=np.float)
            FP = np.sum(self.metrics_strict["FP"], dtype=np.float)
            FN = np.sum(self.metrics_strict["FN"], dtype=np.float)

        # updating accuracy (micro averaged) -> (TP + TN) / (TP + TN + FP + FN)
        self.accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        # updating precision -> TP / (TP + FP)
        self.precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # updating recall -> TP / (TP + FN)
        self.recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

        # updating F1 measure
        pr  = self.precision * self.recall
        p_r = self.precision + self.recall
        self.F1 = 2 * pr / p_r if p_r > 0 else 0

        # returning metrics
        return self.accuracy, self.precision, self.recall, self.F1

    def get_f1_label(self, label, greedy=False, update_metrics=False):
        """
        This utility can be used to retrieve F1 measure for a single label
        :param label: string, label to retrieve F1 for in the format xxx as specified in macros@LABELS 
        :param greedy: (optional) boolean, wheter or not return the greedy metrics
        :param update_metrics: (optional) boolean, wheter or not update metrics
        :return: tuple, (F1 measure, number of elements)
        """
        # updating metrics if necessary
        if update_metrics:
            self.compute_metrics()

        # retrieving label id
        lab_id = LABELS.index(label)

        # computing TP, FP and FN
        if greedy:
            TP = self.metrics_greedy["TP"][lab_id]
            FP = self.metrics_greedy["FP"][lab_id]
            FN = self.metrics_greedy["FN"][lab_id]
        else:
            TP = self.metrics_strict["TP"][lab_id]
            FP = self.metrics_strict["FP"][lab_id]
            FN = self.metrics_strict["FN"][lab_id]

        # computing F1 measure for label
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {"F1" : F1, "precision" : precision, "recall" : recall}


    @staticmethod
    def write_log_metric(metric, filepath, label=""):
        """
        This utility can be used to write a metric object from a model evaluation to a file
        :param metric: dictionary as follow 
            {"F1"        : -, 
             "accuracy"  : -, 
             "precision" : -, 
             "recall"    : -, 
             "F1Labels"  : {"F1" : -, "precision" : -, "recall" : -} }

        :param filepath: string, the file where to write
        :param label: string, label to add to log 
        :return: None
        """
        with open(filepath, "a") as metrics_file:

            F1Labels = metric["F1Labels"]

            metrics_file.write("\n\n" + str(label) + "\n")

            for lab in F1Labels:
                metrics_file.write("{} : F1 {} - precision {} - recall {}\n".format(lab, F1Labels[lab]["F1"],
                                                                                         F1Labels[lab]["precision"],
                                                                                         F1Labels[lab]["recall"]))
            metrics_file.write("F1 micro avg  -> {}\n".format(metric["F1"]))
            metrics_file.close()    

    @staticmethod
    def write_log_metric_nfold(metric_list, N, filepath):
        """
        This utility can be used to write a list of metric object from a n-fold model evaluation to a file
        :param metric: dictionary as follow 
            List : {"F1"        : -, 
                    "accuracy"  : -, 
                    "precision" : -, 
                    "recall"    : -, 
                    "F1Labels"  : {"F1" : -, "precision" : -, "recall" : -} }

        :param N: integer 
        :param filepath: string, the file where to write
        :return: None
        """
        # instantiating metrics
        accuracy = 0.0
        F1       = 0.0
        labels_m = {"PER" : {"F1" : 0.0, "precision" : 0.0, "recall" : 0.0},
                    "ORG" : {"F1" : 0.0, "precision" : 0.0, "recall" : 0.0},
                    "LOC" : {"F1" : 0.0, "precision" : 0.0, "recall" : 0.0}}
        # micro-averaging the metrics
        for metric in metric_list:
            accuracy += (metric["accuracy"] / N)
            F1       += (metric["F1"] / N)
            for l in labels_m: 
                for m in labels_m[l]:
                    labels_m[l][m] += (metric["F1Labels"][l][m] / N)
            
        # writing out metrics
        with open(filepath, "a") as metrics_file:
            metrics_file.write("\n\n\n")
            metrics_file.write("ACC     : {}\n".format(accuracy))
            metrics_file.write("F1      : {}\n".format(F1))
            for l in labels_m:
                metrics_file.write("{} : F1 {} - precision {} - recall {}\n".format(l, labels_m[l]["F1"],
                                                                                       labels_m[l]["precision"],
                                                                                       labels_m[l]["recall"]))
            metrics_file.close()


if __name__ == "__main__":
    labelled = [[0, 1, 2, 0, 0, 0, 1, 2, 3]]
    outputd  = [[0, 0, 0, 0, 0, 0, 1, 2, 3]]
    S = SimpleMetrics(labelled, outputd)