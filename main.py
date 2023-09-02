from DGAFramework.Framework import Framework
from classifiers.common.datasetManagerCommon import DatasetManagerCommon

from classifiers.Yu.yuLSTM import YuLSTM
from classifiers.Yu.yuCNN import YuCNN
from classifiers.Yang.yangLSTM import YangLSTM
from classifiers.Yang.yangCNN import YangCNN
from classifiers.Woodbridge.woodbridgeLSTM import WoodbridgeLSTM
from classifiers.Yu.tweet2Vec import Tweet2Vec
from classifiers.Yu.zhang import Zhang
from classifiers.Yu.expose import Expose
from classifiers.Yu.tweet2Vec2 import Tweet2Vec2
from classifiers.Yu.baseline import Baseline
from classifiers.Yu.mlp import MLP
from classifiers.Berman.CNNMaxPooling import CNNMaxPooling
from classifiers.Berman.BermanLSTM import BermanLSTM
from classifiers.Berman.BermanCNNLSTM import BermanCNNLSTM
from classifiers.Berman.BermanBidirectional import BermanBidirectional
from classifiers.Vinayakumar.DBD import DBD

PATH_TRAIN_DGA = "datasets/final/dgaTrain152000.csv"
PATH_TEST_DGA = "datasets/final/dgaTest152000.csv"
PATH_VALIDATION_DGA = "datasets/final/dgaValidation152000.csv"
PATH_NON_DGA = "datasets/final/NONdga152000.csv"

# Create Framework
framework = Framework()

# Create DatasetManager1 implementation
datasetManager = DatasetManagerCommon()

# Set in framework the dataset to use
framework.defineDatasetManager(datasetManager)

# Add dga dataset
framework.addTrainDataset(PATH_TRAIN_DGA)
framework.addValidationDataset(PATH_VALIDATION_DGA)
framework.addTestDataset(PATH_TEST_DGA)

# Add Non dga dataset
framework.addDataset(PATH_NON_DGA, False)

classifiers = [YuLSTM,
               YuCNN,
               YangLSTM,
               YangCNN,
               WoodbridgeLSTM,
               Tweet2Vec,
               Zhang,
               Expose,
               Tweet2Vec2,
               Baseline,
               MLP,
               CNNMaxPooling,
               BermanLSTM,
               BermanCNNLSTM,
               BermanBidirectional,
               DBD,
               ]

for classifier in classifiers:
    classifierObj = classifier()
    framework.addClassifier(classifierObj)

results = framework.getResults()

# Train all classifiers
framework.train()

# Test all classifiers
framework.test()

results = framework.getResults()

filename = "./results/general.csv"
f = open(filename, "a")

separator = ";"
f.write("model" + separator + framework.getResultByIndex(0).toCSVheader(separator))
f.write("\n")

for i in range(len(results)):
    print(framework.getClassifierByIndex(i).__class__.__name__)
    print(framework.getResultByIndex(i).toString())
    
    f.write(framework.getClassifierByIndex(i).__class__.__name__ + separator + framework.getResultByIndex(i).toCSV(separator))
    f.write("\n")

f.close()