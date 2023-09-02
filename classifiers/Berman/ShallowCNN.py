from DGAFramework.Classifier.Classifier import Classifier
from DGAFramework.Result.Result import Result
from DGAFramework.DataElement.DataElement import DataElement

from classifiers.common.resultCommon import ResultCommon
from classifiers.common.commonData import CommonData
from classifiers.common.TimeRecord import TimeRecord
from classifiers.common.history import saveHistory

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, Conv2D, Flatten
from tensorflow.keras import Model
import tensorflow as tf
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy

class ShallowCNN(Classifier):

    model = None
    lstm_size = 50
    epochs = CommonData.epochs
    patience = 100
    batch_size = CommonData.batch_size
    activation = 'softsign'
    model_name = "shallowCNN"
    maxlen = CommonData.maxlen

    save_file = "./classifiers/models/"+model_name+".h5"

    charset = None
    dictionary = None
    reverse_dictionary = None

    def __init__(self) -> None:

        self.charset = list("abcdefghijklmnopqrstuvwxyz0123456789.-")
        self.dictionary = dict(zip(self.charset, range(len(self.charset))))
        self.reverse_dictionary = dict(zip(range(len(self.charset)), self.charset))

        self.model = Sequential (name=self.model_name)
        #self.model.add(Embedding(128))
        self.model.add(Conv2D(filters = 1000, kernel_size = 2, padding = 'same',
                              kernel_initializer='glorot_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(100, kernel_initializer='glorot_normal'))
        self.model.add(Dense(1, kernel_initializer='glorot_normal'))

        self.model.compile(loss ='binary_crossentropy',
        optimizer='adam', metrics=CommonData.metrics)
        
        #self.model.summary()
        
    def domain_to_vector(self, domain):
        res = []
        for c in list(domain):
            v = [float(0)] * len(self.dictionary)
            v[self.dictionary[c]] = 1.0
            res.append(v)
        return res

    def train(self, train:set, validation:set):

        x = []
        y = []

        for dataelement in train:
            x.append(self.domain_to_vector(dataelement.domain))
            if dataelement.isDGA:
                y.append(1)
            else:
                y.append(0)

        x_val = []
        y_val = []

        for dataelement in validation:
            x_val.append(self.domain_to_vector(dataelement.domain))
            if dataelement.isDGA:
                y_val.append(1)
            else:
                y_val.append(0)

        novalue = [float(0)] * len(self.dictionary)
        x_noarray_pad = pad_sequences(x, dtype=float, value=novalue,
                                      padding='post', maxlen=self.maxlen)

        x = numpy.array(x_noarray_pad, dtype=float)
        y = numpy.array(y, dtype=float)

        novalue = [float(0)] * len(self.dictionary)
        x_noarray_pad = pad_sequences(x_val, dtype=float, value=novalue,
                                      padding='post', maxlen=self.maxlen)

        x_val = numpy.array(x_noarray_pad, dtype=float)
        y_val = numpy.array(y_val, dtype=float)

        checkpoint=ModelCheckpoint(self.save_file, monitor='val_accuracy',
                                   verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto')
        timeCallback = TimeRecord()
        history = self.model.fit(x, y, epochs=self.epochs,
                                 verbose = CommonData.verbose,
                                 validation_data=(x_val, y_val),
                                 batch_size=self.batch_size,
                                 callbacks=[checkpoint,timeCallback])
        self.times = timeCallback.times

        history_file = "./classifiers/history/"+self.model_name+".hys"
        saveHistory(history, history_file)


    def test(self, test:set) -> Result:
        
        best_model = load_model(self.save_file)

        x_test = []
        y_test = []

        for dataelement in test:
            x_test.append(self.domain_to_vector(dataelement.domain))
            if dataelement.isDGA:
                y_test.append(1)
            else:
                y_test.append(0)  

        novalue = [float(0)] * len(self.dictionary)
        x_noarray_pad = pad_sequences(x_test, dtype=float, value=novalue,
                                      padding='post', maxlen=self.maxlen)

        x_test = numpy.array(x_noarray_pad, dtype=float)
        y_test = numpy.array(y_test, dtype=float)
        
        scores = best_model.evaluate(x_test, y_test, batch_size=10)
        #print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        return ResultCommon(scores[1]*100, scores[2]*100, scores[3]*100, scores[4], scores[5], scores[6], scores[7], scores[8], self.times)
    
    def predict(self, dataElement:DataElement):
        best_model = load_model(self.save_file)
        x = []
        dataVec = []
        for character in dataElement.domain:
            dataVec.append(ord(character) - 33) # 33 is the first printeable char (is !)
        x.append(dataVec)
        x = pad_sequences(x, padding='post', maxlen=self.maxlen)
        x = numpy.array(x, dtype=float)

        y = best_model.predict(x)[0][0]

        return y
        
    def predicts(self, dataElements:list):
        best_model = load_model(self.save_file)
        x = []
        dataVec = []
        for dataelement in dataElements:
            dataVec = []
            for character in dataelement.domain:
                    dataVec.append(ord(character) - 33) # 33 is the first printeable char (is !)
            x.append(dataVec)
        x = pad_sequences(x, padding='post', maxlen=self.maxlen)
        x = numpy.array(x, dtype=float)

        y = best_model.predict(x)

        return y