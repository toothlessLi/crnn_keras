from keras.layers.core import Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD
import keras.backend as K
import warpctc_tensorflow


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(nb_classes, image_size, max_label_length):
    basemodel, y_pred, model_input = crnn(nb_classes, image_size)
    labels = Input(name='the_labels', shape=[max_label_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    opt = Adam(lr=0.001, clipnorm=10.0)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])
    return model


def crnn(nb_classes, input_shape=(32, 280, 3)):
    model_input = Input(shape=input_shape, name='the_input')

    m = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)
    # h/2, w/2

    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)
    # h/4, w/4

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu')(m)
    # h/4 - 2, w/4

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu')(m)
    # h/4 - 4, w/4

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu')(m)
    # h/4 - 6, w/4

    m = Permute((2, 1, 3))(m)
    m = Reshape(target_shape=(-1, 1, (input_shape[0]//4 - 6)*256))(m)
    m = TimeDistributed(Flatten())(m)
    # m = Bidirectional(SimpleRNN(256, return_sequences=True, dropout=0.5), name='rnn1')(m)
    m = Dense(1024, activation='relu')(m)
    m = Dropout(0.5)(m)
    y_pred = Dense(nb_classes, activation='softmax')(m)

    basemodel = Model(inputs=model_input, outputs=y_pred)
    basemodel.summary()
    return basemodel, y_pred, model_input
