#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           May 29, 2023  ##
#############################################

import warnings

from DGAFramework.Classifier.Classifier import Classifier
from DGAFramework.DataElement.DataElement import DataElement
from DGAFramework.Result.Result import Result
from DGAFramework.DatasetManager.DatasetManager import DatasetManager

################################################################################
#  CONSTANT STRINGS  ###########################################################
################################################################################

WRONG_INDEX_CLASSIFIERS = '''\n  ERROR:

  Index out of bounds...

  Number of classifiers: {total}
  Posible index values: 0 - {posibleIndex}
  Index: {currentIndex}
'''

WRONG_INDEX_RESULTS = '''\n  ERROR:

  Index out of bounds...

  Number of results: {total}
  Posible index values: 0 - {posibleIndex}
  Index: {currentIndex}
'''

WARNING_INDEX_CLASSIFIER_EXIST = '''\n  WARNING:

  Does not exist the classifier
'''

ERROR_NO_DEBUG = '''\n  ERROR:

  Debug mode is NOT activated

  Not allowed to execute {nameFunction} without debug mode
'''

################################################################################
#  CODE  #######################################################################
################################################################################

class Framework:

    debug = False
    datasetManager:DatasetManager = None
    classifiers = []
    results = []

    def __init__(self, debugMode:bool = False):
        self.debug = debugMode
        self.datasetManager = None
        self.classifiers.clear()
        self.results.clear()
        if self.debug:
            print("\n#############################################")
            print("########### DEBUG MODE ACTIVATED ############")
            print("#############################################\n")

    def defineDatasetManager(self, datasetManager:DatasetManager):
        """
        Define the instance of a DatasetManager that will be used in
        the framework as dataset manager
        
        :param datasetManager: DatasetManager that will be used
        """
        self.datasetManager = datasetManager

    def addDataset(self, path:str, randomSets:bool):
        """
        Add DataElements splitted in train, validation and test sets
        from the specified file and put it into DatasetManager

        :param path: Path from DataElements are being read
        :param randomSets: Determinates if sets will be randomized
        """
        self.datasetManager.add(path, randomSets)
        if self.debug:
            print("#############################################")
            print("###### NEW DataElements added to sets #######")
            print("#############################################\n")
            print("  new size of TRAIN set     : ", len(self.datasetManager.getTrain()))
            print("  new size of VALIDATION set: ", len(self.datasetManager.getValidation()))
            print("  new size of TEST set      : ", len(self.datasetManager.getTest()), "\n")

    def addTrainDataset(self, path:str):
        """
        Adds DataElements to train set
        from the specified file

        :param path: Path from DataElements are being read
        """
        self.datasetManager.addTrain(path)

    def addValidationDataset(self, path:str):
        """
        Adds DataElements to validation set
        from the specified file

        :param path: Path from DataElements are being read
        """
        self.datasetManager.addValidation(path)

    def addTestDataset(self, path:str):
        """
        Adds DataElements to test set
        from the specified file

        :param path: Path from DataElements are being read
        """
        self.datasetManager.addTest(path)

    def addClassifier(self, classifier:Classifier):
        """
        Adds a classifier to the framework list to run

        :param classifier: Classifier to add
        """
        # Adds the classifier to the list of classifiers
        self.classifiers.append(classifier)
        # Adds None to results in the same index
        # of the previous added classifier
        self.results.append(None)

    def getClassifierByIndex(self, index:int) -> Classifier:
        """
        Returns the classifier by the specified index
        
        :param index: Index of the classifier to return
        :return: The classifier of the specified index
        """
        if index >= len(self.classifiers) or index < 0:
            raise Exception(WRONG_INDEX_CLASSIFIERS.format(
                total = len(self.classifiers),
                posibleIndex = len(self.classifiers)-1,
                currentIndex = index
            ))
        return self.classifiers[index]

    def clearClassifiers(self):
        """
        Clear all classifiers (also results)
        """
        self.classifiers.clear()
        self.results.clear()

    def run(self):
        """
        Train and test all classifiers
        """
        i = 0
        while i < len(self.classifiers):
            self.trainByIndex(i)
            self.testByIndex(i)
            i = i+1


    def runC(self, classifier:Classifier):
        """
        Train and test the classifier, must be in the classifier list
        
        :param classifier: Classifier to train and test
        """
        index = -1
        try:
            index = self.classifiers.index(classifier)
        except ValueError:
            warnings.warn(WARNING_INDEX_CLASSIFIER_EXIST)
            return None
        self.runByIndex(index)

    def runByIndex(self, index:int):
        """
        Train and test the classifier specified by index
        
        :param index: Index of the classifier to train and test
        """
        if index >= len(self.classifiers) or index < 0:
            raise Exception(WRONG_INDEX_CLASSIFIERS.format(
                total = len(self.classifiers),
                posibleIndex = len(self.classifiers)-1,
                currentIndex = index
            ))
        self.trainByIndex(index)
        self.testByIndex(index)

    def train(self):
        """
        Train all classifiers
        """
        i = 0
        while i < len(self.classifiers):
            self.trainByIndex(i)
            i = i+1

    def trainC(self, classifier:Classifier):
        """
        Train the classifier, must be in the classifier list
        
        :param classifier: Classifier to train
        """
        index = -1
        try:
            index = self.classifiers.index(classifier)
        except ValueError:
            warnings.warn(WARNING_INDEX_CLASSIFIER_EXIST)
            return None
        self.trainByIndex(index)

    def trainByIndex(self, index:int):
        """
        Train the classifier specified by index
        
        :param index: Index of the classifier to train
        """
        if index >= len(self.classifiers) or index < 0:
            raise Exception(WRONG_INDEX_CLASSIFIERS.format(
                total = len(self.classifiers),
                posibleIndex = len(self.classifiers)-1,
                currentIndex = index
            ))
        self.classifiers[index].train(self.datasetManager.getTrain(),
                                      self.datasetManager.getValidation())

    def test(self):
        """
        Tests all classifiers
        """
        i = 0
        while i < len(self.classifiers):
            self.testByIndex(i)
            i = i+1

    def testC(self, classifier:Classifier):
        """
        Tests the classifier, must be in the classifier list
        
        :param classifier: Classifier to test
        """
        index = -1
        try:
            index = self.classifiers.index(classifier)
        except ValueError:
            warnings.warn(WARNING_INDEX_CLASSIFIER_EXIST)
            return None
        self.testByIndex(index)

    def testByIndex(self, index:int):
        """
        Tests the classifier specified by index
        
        :param index: Index of the classifier to test
        """
        if index >= len(self.classifiers) or index < 0:
            raise Exception(WRONG_INDEX_CLASSIFIERS.format(
                total = len(self.classifiers),
                posibleIndex = len(self.classifiers)-1,
                currentIndex = index
            ))
        self.results[index] = self.classifiers[index].test(self.datasetManager.getTest())

    def getResults(self) -> list:
        """
        Return a list with all results
        
        :return: Return a list with all results (if the classifier
                is not tested, return None in that cases)
        """
        return self.results

    def getResultC(self, classifier:Classifier) -> Result:
        """
        Returns the result by the specified classifier
        
        :param index: Index of the result to return
        :return: Result of the classifier, if is not already
                tested or classifier not exists, will return None
        """
        index = -1
        try:
            index = self.classifiers.index(classifier)
        except ValueError:
            warnings.warn(WARNING_INDEX_CLASSIFIER_EXIST)
            return None
        return self.getResultByIndex(index)

    def getResultByIndex(self, index:int) -> Result:
        """
        Returns the result by the specified index
        
        :param index: Index of the result to return
        :return: Result of the index, if is not already
                tested, will return None
        """
        if index >= len(self.results) or index < 0:
            raise Exception(WRONG_INDEX_RESULTS.format(
                total = len(self.results),
                posibleIndex = len(self.results)-1,
                currentIndex = index
            ))
        return self.results[index]

################################################################################
#  DEBUG MODE FUNCTIONS  #######################################################
################################################################################
    def printAllDataset(self):
        
        # Check if is executing this method in debug mode
        if not self.debug:
            print(ERROR_NO_DEBUG.format(nameFunction = "printAllDataset()"))
            exit(1)

        print("#############################################")
        print("############## TRAIN SET LIST ###############")
        print("#############################################\n")

        for dataelement in self.datasetManager.getTrain():
            print("  - ", dataelement.domain, " -> ", dataelement.isDGA)
        print()

        print("#############################################")
        print("########### VALIDATION SET LIST #############")
        print("#############################################\n")

        for dataelement in self.datasetManager.getValidation():
            print("  - ", dataelement.domain, " -> ", dataelement.isDGA)
        print()

        print("#############################################")
        print("############### TEST SET LIST ###############")
        print("#############################################\n")

        for dataelement in self.datasetManager.getTest():
            print("  - ", dataelement.domain, " -> ", dataelement.isDGA)
        print()