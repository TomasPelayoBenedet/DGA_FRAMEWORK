import pytest

from DGAFramework.Framework import Framework
from test.classes.frameworkTest.datasetManagerFrameworkTest import DatasetManagerFrameworkTest
from test.classes.frameworkTest.simpleClassifierFrameworkTest import SimpleClassifierFrameworkTest
from test.classes.frameworkTest.simpleClassifierFrameworkTest2 import SimpleClassifierFrameworkTest2
from test.classes.frameworkTest.simpleClassifierFrameworkTest3 import SimpleClassifierFrameworkTest3


PATH_DGA = "test/datasets/datasetManagerTest/dga.txt"
PATH_NON_DGA = "test/datasets/datasetManagerTest/NOdga.txt"

def test_run():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    framework.run()

    for result in framework.getResults():
        assert result != None

def test_get_classifier():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    assert framework.getClassifierByIndex(0) == classifier1
    assert framework.getClassifierByIndex(1) == classifier2
    assert framework.getClassifierByIndex(2) == classifier3


def test_clear_classifiers():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    framework.clearClassifiers()

    assert len(framework.classifiers) == 0

def test_train_test():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    framework.train()

    for result in framework.getResults():
        assert result == None

    framework.test()

    for result in framework.getResults():
        assert result != None

def test_C_run():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    for result in framework.getResults():
        assert result == None

    framework.runC(classifier1)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) == None
    assert framework.getResultC(classifier3) == None

    framework.runC(classifier2)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) != None
    assert framework.getResultC(classifier3) == None

    framework.runC(classifier3)

    for result in framework.getResults():
        assert result != None

def test_C_train_test():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    for result in framework.getResults():
        assert result == None

    framework.trainC(classifier1)

    assert framework.getResultC(classifier1) == None
    assert framework.getResultC(classifier2) == None
    assert framework.getResultC(classifier3) == None

    framework.testC(classifier1)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) == None
    assert framework.getResultC(classifier3) == None

    framework.trainC(classifier2)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) == None
    assert framework.getResultC(classifier3) == None

    framework.testC(classifier2)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) != None
    assert framework.getResultC(classifier3) == None

    framework.trainC(classifier3)

    assert framework.getResultC(classifier1) != None
    assert framework.getResultC(classifier2) != None
    assert framework.getResultC(classifier3) == None

    framework.testC(classifier3)

    for result in framework.getResults():
        assert result != None

def test_index_run():
    framework = Framework()
    datasetManager = DatasetManagerFrameworkTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)

    classifier1 = SimpleClassifierFrameworkTest()
    framework.addClassifier(classifier1)

    classifier2 = SimpleClassifierFrameworkTest2()
    framework.addClassifier(classifier2)

    classifier3 = SimpleClassifierFrameworkTest3()
    framework.addClassifier(classifier3)

    for result in framework.getResults():
        assert result == None

    framework.runByIndex(0)

    assert framework.getResultByIndex(0) != None
    assert framework.getResultByIndex(1) == None
    assert framework.getResultByIndex(2) == None

    framework.runByIndex(1)

    assert framework.getResultByIndex(0) != None
    assert framework.getResultByIndex(1) != None
    assert framework.getResultByIndex(2) == None

    framework.runByIndex(2)

    for result in framework.getResults():
        assert result != None