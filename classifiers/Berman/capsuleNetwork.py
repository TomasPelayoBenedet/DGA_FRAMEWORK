from DGAFramework.Classifier.Classifier import Classifier
from DGAFramework.Result.Result import Result
from DGAFramework.DataElement.DataElement import DataElement

from classifiers.common.resultCommon import ResultCommon
from classifiers.common.commonData import CommonData
from classifiers.common.TimeRecord import TimeRecord
from classifiers.common.history import saveHistory

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SpatialDropout1D, Dense, Input, LSTM, BatchNormalization, MaxPooling1D, Embedding, Conv1D, Dropout, Activation, Bidirectional, Conv2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras import layers
from keras import backend as K
import tensorflow as tf
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

class CapsuleNetwork(Classifier):

    model = None
    lstm_size = 50
    epochs = CommonData.epochs
    patience = 100
    batch_size = CommonData.batch_size
    activation = 'softsign'
    model_name = "capsuleNetwork"
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
        self.model.add(Conv1D(filters=256, kernel_size = 8, padding='valid'))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Conv1D(filters=512, kernel_size = 4, padding='valid'))
        self.model.add(Dropout(0.7))
        self.model.add(Conv1D(filters=256, kernel_size = 4, padding='valid'))
        self.model.add(PrimaryCap(dim_capsule=8, n_channels = 32,
                                   kernel_size = 4, strides = 2, padding='valid'))
        self.model.add(CapsuleLayer(num_capsule = 1, dim_capsule = 16, routing = 7))
        self.model.add(Length(0.85, 0.15))

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