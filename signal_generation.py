import numpy as np


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
             rsi_signals(current_data),
             mfi_signals(current_data),
             macd_signals(current_data),
             buy_mode]

    return np.asarray(state).reshape(5, )
