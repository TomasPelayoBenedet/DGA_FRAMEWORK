#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           May 29, 2023  ##
#############################################

from DGAFramework.Result.Result import Result

import math
import statistics

class ResultCommon(Result):

    def __init__(self, accuracy, precision, recall, fp, fn, tp, tn, auc, times) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        if precision + recall != 0:
            self.f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            self.f1_score = 0
        
        if fp + tn != 0:
            self.fpr = fp / (fp+tn)
        else:
            self.fpr = 0

        if tp + fn != 0:
            self.tpr = tp/(tp+fn)
        else:
            self.tpr = 0
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.tn = tn
        self.auc = auc
        if math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) != 0:
            self.mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:
            self.mcc = 0
        
        # Kappa implemented by
        # https://stackoverflow.com/questions/58735642/how-to-correctly-implement-cohen-kappa-metric-in-keras

        if (tn+fp+fn+tp) != 0:
            p0 = (tn+tp)/(tn+fp+fn+tp)
            pa = ((tn+tp)/(tn+fp+fn+tp))*((tn+fn)/(tn+fp+fn+tp))
            pb = ((fn+tp)/(tn+fp+fn+tp))*((fp+tp)/(tn+fp+fn+tp))
            pe = pa + pb
            self.kappa = (p0-pe)/(1-pe)
        else:
            self.kappa = 0

        self.time = statistics.mean(times)

    def toString(self) -> str:
        ret = " * Accuracy   -> " + str(self.accuracy) + "\n"
        ret = ret + " * Precision  -> " + str(self.precision) + "\n"
        ret = ret + " * Recall     -> " + str(self.recall) + "\n"
        ret = ret + " * F1 score   -> " + str(self.f1_score) + "\n"
        ret = ret + " * FPR        -> " + str(self.fpr) + "\n"
        ret = ret + " * TPR        -> " + str(self.tpr) + "\n"
        ret = ret + " * AUC        -> " + str(self.auc) + "\n"
        ret = ret + " * MCC        -> " + str(self.mcc) + "\n"
        ret = ret + " * Kappa      -> " + str(self.kappa) + "\n"
        ret = ret + " * Epoch time -> " + str(self.time)
        return ret
    
    def toCSVheader(self, separator:str) -> str:
        ret = "accuracy" + separator + "precision" + separator + "recall"
        ret = ret + separator + "f1" + separator + "fpr" + separator + "tpr"
        ret = ret + separator + "auc" + separator + "fp"  + separator + "fn"
        ret = ret + separator + "tp" + separator + "tn" + separator + "mcc"
        ret = ret + separator + "kappa"
        ret = ret + separator + "time"
        return ret

    def toCSV(self, separator:str) -> str:
        ret = str(self.accuracy) + separator + str(self.precision)
        ret = ret + separator + str(self.recall) + separator
        ret = ret + str(self.f1_score) + separator + str(self.fpr) + separator
        ret = ret + str(self.tpr) + separator + str(self.auc) + separator
        ret = ret + str(self.fp) + separator + str(self.fn) + separator
        ret = ret + str(self.tp) + separator + str(self.tn) + separator
        ret = ret + str(self.mcc) + separator + str(self.kappa)
        ret = ret + separator + str(self.time)
        return ret