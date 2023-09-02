#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           Aug 04, 2023  ##
#############################################

import warnings

from DGAFramework.DataElement.DataElement import DataElement

from classifiers.Yu.yuLSTM import YuLSTM
from classifiers.Woodbridge.woodbridgeLSTM import WoodbridgeLSTM
from classifiers.Yu.tweet2Vec import Tweet2Vec
from classifiers.Yu.tweet2Vec2 import Tweet2Vec2
from classifiers.Yu.zhang import Zhang
from classifiers.Yu.expose import Expose
from classifiers.Vinayakumar.DBD import DBD

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

################################################################################
#  CONSTANT STRINGS  ###########################################################
################################################################################

FILENAME_MODEL="models/apirest_lr.h5"

################################################################################
#  CODE  #######################################################################
################################################################################

class APIrest:

    def __init__(self, debugMode:bool = False):
        self.debug = debugMode
        self.classifiers = [YuLSTM(), WoodbridgeLSTM(), Tweet2Vec(), Tweet2Vec2(),
                            Zhang(), Expose(), DBD()]
        if os.path.isfile(FILENAME_MODEL):
            self.model = pickle.load(open(FILENAME_MODEL, 'rb'))
            print("Loaded model!")
        else:
            self.model = LogisticRegression(solver='liblinear', random_state=0)
            
        if self.debug:
            print("\n#############################################")
            print("########### DEBUG MODE ACTIVATED ############")
            print("#############################################\n")

    def predict(self, dataElement:DataElement) -> float:
        """
        Explanation
        
        :param dataElement: Explanation
        """
        x = []
        x.append([])

        for classifier in self.classifiers:
            x[0].append(classifier.predict(dataElement))

        return self.logisticRegression(x)

    def logisticRegression(self, predictions:list) -> float:
        x = self.model.predict_proba(predictions)
        return x[0][1]

    def trainLogisticRegression(self, train:list, validation:list, override=False):
        if os.path.isfile(FILENAME_MODEL) and not override:
            return
        # Else, train
        # Train
        x = []
        y = []

        x_aux = []
        for classifier in self.classifiers:
            x_aux.append(classifier.predicts(train))

        first_row = True
        for x_sample in x_aux:
            i = 0
            while i < len(x_sample):
                if first_row:
                    x.append([])
                    x[i].append(x_sample[i][0])
                else:
                    x[i].append(x_sample[i][0])
                i = i+1
            first_row = False

        for sample in train:
            if sample.isDGA:
                y.append(1)
            else:
                y.append(0)

        self.model.fit(x,y)
        
        # Validation

        x_val = []
        y_val = []

        x_aux = []
        for classifier in self.classifiers:
            x_aux.append(classifier.predicts(validation))

        first_row = True
        for x_sample in x_aux:
            i = 0
            while i < len(x_sample):
                if first_row:
                    x_val.append([])
                    x_val[i].append(x_sample[i][0])
                else:
                    x_val[i].append(x_sample[i][0])
                i = i+1
            first_row = False

        for sample in validation:
            if sample.isDGA:
                y_val.append(1)
            else:
                y_val.append(0)

        score_ = self.model.score(x_val, y_val)

        print(score_)

        pickle.dump(self.model, open(FILENAME_MODEL, 'wb'))


    def predictSome(self, dataElements:list) -> list:
        x = []

        x_aux = []
        
        for classifier in self.classifiers:
            x_aux.append(classifier.predicts(dataElements))

        first_row = True
        for x_sample in x_aux:
            i = 0
            while i < len(x_sample):
                if first_row:
                    x.append([])
                    x[i].append(x_sample[i][0])
                else:
                    x[i].append(x_sample[i][0])
                i = i+1
            first_row = False
        
        return self.logisticRegressionSome(x)

    def logisticRegressionSome(self, predictions:list) -> list:
        return self.model.predict_proba(predictions)