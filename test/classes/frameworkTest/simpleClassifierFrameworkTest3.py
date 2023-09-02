from DGAFramework.Classifier.Classifier import Classifier
from DGAFramework.Result.Result import Result

from test.classes.frameworkTest.resultFrameworkTest import ResultFrameworkTest

import time

import numpy


class SimpleClassifierFrameworkTest3(Classifier):

    def __init__(self) -> None:
        pass
        
    def train(self, train:set, validation:set):

        # Train stuff
        time.sleep(0.5)

    def test(self, test:set) -> Result:

        return ResultFrameworkTest(50)