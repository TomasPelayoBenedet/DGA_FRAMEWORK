import pytest

from DGAFramework.Framework import Framework
from test.classes.datasetManagerTest.datasetManagerDatasetManagerTest import DatasetManagerDatasetManagerTest

PATH_DGA = "test/datasets/datasetManagerTest/dga.txt"
PATH_NON_DGA = "test/datasets/datasetManagerTest/NOdga.txt"

def test_load_random():
    framework = Framework()
    datasetManager = DatasetManagerDatasetManagerTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)
    a = []
    for x in datasetManager.getTest():
        a.append(x.domain)
    
    framework.datasetManager.clear()

    framework.addDataset(PATH_DGA, True)
    framework.addDataset(PATH_NON_DGA, True)
    b = []
    for x in datasetManager.getTest():
        b.append(x.domain)

    assert a != b

def test_load_not_random():
    framework = Framework()
    datasetManager = DatasetManagerDatasetManagerTest()
    framework.defineDatasetManager(datasetManager)
    framework.addDataset(PATH_DGA, False)
    framework.addDataset(PATH_NON_DGA, False)
    a = []
    for x in datasetManager.getTest():
        a.append(x.domain)
    
    framework.datasetManager.clear()

    framework.addDataset(PATH_DGA, False)
    framework.addDataset(PATH_NON_DGA, False)
    b = []
    for x in datasetManager.getTest():
        b.append(x.domain)

    assert a != b

def test_set_other_splits():
    framework = Framework()
    datasetManager = DatasetManagerDatasetManagerTest()
    framework.defineDatasetManager(datasetManager)
    datasetManager.setPercentages(100, 0, 0)
    framework.addDataset(PATH_DGA, False)
    
    assert len(framework.datasetManager.getTrain()) == 32000
    assert len(framework.datasetManager.getValidation()) == 0
    assert len(framework.datasetManager.getTest()) == 0

    datasetManager.clear()
    datasetManager.setPercentages(40, 30, 30)
    framework.addDataset(PATH_DGA, False)

    assert len(framework.datasetManager.getTrain()) == 12800
    assert len(framework.datasetManager.getValidation()) == 9600
    assert len(framework.datasetManager.getTest()) == 9600

def test_add_manually():
    framework = Framework()
    datasetManager = DatasetManagerDatasetManagerTest()
    framework.defineDatasetManager(datasetManager)
    datasetManager.setPercentages(100, 0, 0)

    framework.addTrainDataset(PATH_DGA)
    framework.addTrainDataset(PATH_DGA)

    assert len(framework.datasetManager.getTrain()) == 64000
    assert len(framework.datasetManager.getValidation()) == 0
    assert len(framework.datasetManager.getTest()) == 0

    framework.addValidationDataset(PATH_DGA)
    framework.addValidationDataset(PATH_DGA)
    framework.addValidationDataset(PATH_DGA)

    assert len(framework.datasetManager.getTrain()) == 64000
    assert len(framework.datasetManager.getValidation()) == 96000
    assert len(framework.datasetManager.getTest()) == 0

    framework.addTestDataset(PATH_DGA)
    framework.addTestDataset(PATH_DGA)
    framework.addTestDataset(PATH_DGA)
    framework.addTestDataset(PATH_DGA)

    assert len(framework.datasetManager.getTrain()) == 64000
    assert len(framework.datasetManager.getValidation()) == 96000
    assert len(framework.datasetManager.getTest()) == 128000
