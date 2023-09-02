from tensorflow.keras.metrics import Precision, Recall, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, AUC

class CommonData:
    epochs = 50
    maxlen=70 # Maxium domain name lenght
    batch_size = 25
    verbose = 1
    #verbose = 0 # No output
    metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives(), TruePositives(), TrueNegatives(), AUC()]