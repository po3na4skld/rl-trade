from keras.layers import Dense, Activation, InputLayer, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


def awesome_model():
    model = Sequential()

    model.add(InputLayer((1, 20)))

    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128))
    model.add(BatchNormalization())

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = Adam(lr=0.004)
    model.load_weights('AwesomeOscillator.model')
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    global ao_model
    ao_model = model


def awesome_oscillator_prediction(current_data):
    ao_signal = ao_model.predict(current_data[:, 5].reshape(1, 1, 20))
    return ao_signal


def acceleration_signals(current_data):
    sequence = current_data[:, 6]
    if sequence[-3] > 0:
        if (sequence[-1] > sequence[-2] > sequence[-3]) and (sequence[-3] < sequence[-4]):
            return 1
        else:
            return 0
    elif sequence[-4] < 0:
        if (sequence[-1] > sequence[-2] > sequence[-3] > sequence[-4]) and (sequence[-4] < sequence[-5]):
            return 1
        else:
            return 0
    else:
        return 0


def rsi_signals(current_data):
    sequence = current_data[:, 10]
    if sequence[-1] < 40:
        return 1
    else:
        return 0


def mfi_signals(current_data):
    mfi = current_data[:, 11]
    volume = current_data[:, 4]
    if (mfi[-1] > mfi[-2]) and (volume[-1] > volume[-2]):
        return 1
    elif (mfi[-1] < mfi[-2]) and (volume[-1] < volume[-2]):
        return 2
    elif (mfi[-1] > mfi[-2]) and (volume[-1] < volume[-2]):
        return 3
    elif (mfi[-1] < mfi[-2]) and (volume[-1] > volume[-2]):
        return 4


def macd_signals(current_data):
    hist = current_data[:, 14]
    if hist[-1] > hist[-2] > hist[-3]:
        return 1
    else:
        return 0


def generate_signals(current_data, buy_mode):
    state = [acceleration_signals(current_data),
             np.round(awesome_oscillator_prediction(current_data)[0][0]),
             rsi_signals(current_data),
             mfi_signals(current_data),
             macd_signals(current_data),
             buy_mode]

    return np.asarray(state).reshape(6, )


awesome_model()
