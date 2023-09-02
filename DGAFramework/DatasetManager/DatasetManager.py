#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           May 30, 2023  ##
#############################################

import warnings

from DGAFramework.DataElement.DataElement import DataElement
import random

################################################################################
#  CONSTANT STRINGS  ###########################################################
################################################################################

WRONG_PERCENTAGES_MESSAGE = '''ERROR:

Wrong percentages of train, validation and test...

train + validation + test =
{trainWPM} + {validationWPM} + {testWPM} =
{resultWPM}
'''

WARNING_PERCENTAGES_MESSAGE = '''WARNING:

Train percentage should be greater than validation percentage...

train <= validation
{trainWPM} <= {validationWPM}
'''

################################################################################
#  CODE  #######################################################################
################################################################################

class DatasetManager:

    # Sets of trains, validation and test
    trainS = set()
    validationS = set()
    testS = set()

    # Percentage (split) of train, validation and test over 100
    trainP = 60
    validationP = 20
    testP = 20

    def __init__(self):
        self.trainS.clear()
        self.validationS.clear()
        self.testS.clear()
        self.trainP = 80
        self.validationP = 10
        self.testP = 10

    def setPercentages(self, train:int, validation:int, test:int):
        """
        Set percentages of train, validation and test
        
        train + validation + test = 100
        
        :param train: Percentage of DataElements to be train
        :param validation: Percentage of DataElements to be validation
        :param test: Percentage of DataElements to be test
        """
        # Check if parameters have correct percentages (SUM = 100)
        if train + validation + test != 100:
            # Case that SUM != 100: raise an error
            raise Exception(WRONG_PERCENTAGES_MESSAGE.format(
                trainWPM = train,
                validationWPM = validation,
                testWPM = test,
                resultWPM = train + validation + test
            ))
        # Check if train percentage is greater than validation percentage
        if (train <= validation):
            warnings.warn(WARNING_PERCENTAGES_MESSAGE.format(
                trainWPM = train,
                validationWPM = validation
            ))
        
        # Set new percentages
        self.trainP = train
        self.validationP = validation
        self.testP = test

    def getTrain(self) -> set:
        """
        Get training set

        :return: {DataElement}
        """
        return self.trainS

    def getValidation(self) -> set:
        """
        Get validation set

        :return: {DataElement}
        """
        return self.validationS

    def getTest(self) -> set:
        """
        Get testing set

        :return: {DataElement}
        """
        return self.testS
    
    def add(self, path:str, randomSets:bool):
        """
        Add DataElements splitted in train, validation and test sets
        from the specified file

        :param path: Path from DataElements are being read
        :param randomSets: Determinates if sets will be randomized
        """
        # Read in dataGeneral all instances of DataElements
        dataGeneral = []
        with open(path, 'r') as f:
            for line in f:
                dataGeneral.append(self.parseDataElement(line))

        # In case that sets must be with randomization items
        if randomSets:
            random.shuffle(dataGeneral)

        # Init values
        i = 0
        maxNumElements = len(dataGeneral)

        # Get training set
        maxIndexTrain = maxNumElements * self.trainP / 100
        while i < maxIndexTrain:
            self.trainS.add(dataGeneral[i])
            i = i+1
        
        # Get validation set
        maxIndexVal = maxNumElements * (self.trainP + self.validationP) / 100
        while i < maxIndexVal:
            self.validationS.add(dataGeneral[i])
            i = i+1
        
        # Get testing set
        while i < maxNumElements:
            self.testS.add(dataGeneral[i])
            i = i+1

    def addTrain(self, path:str):
        """
        Adds DataElements to train set
        from the specified file

        :param path: Path from DataElements are being read
        """
        with open(path, 'r') as f:
            for line in f:
                self.trainS.add(self.parseDataElement(line))

    def addValidation(self, path:str):
        """
        Adds DataElements to validation set
        from the specified file

        :param path: Path from DataElements are being read
        """
        with open(path, 'r') as f:
            for line in f:
                self.validationS.add(self.parseDataElement(line))

    def addTest(self, path:str):
        """
        Adds DataElements to test set
        from the specified file

        :param path: Path from DataElements are being read
        """
        with open(path, 'r') as f:
            for line in f:
                self.testS.add(self.parseDataElement(line))

    def clear(self):
        """
        Clears all elements from train, validation and test sets
        """
        self.trainS.clear()
        self.validationS.clear()
        self.testS.clear()

    def parseDataElement(self, line:str) -> DataElement:
        '''readDataElement function'''
        pass