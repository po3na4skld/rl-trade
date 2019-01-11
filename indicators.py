import pandas as pd
import numpy as np


def sma(p_c, M):
    sma = p_c.rolling(window=M).mean()
    return sma


def market_facilitation_index(p_h, p_l, volume):
    mfi = (p_h - p_l) / volume
    return mfi


def awesome_oscillator(p_h, p_l, period_1, period_2):
    median_price = (p_h + p_l) / 2
    sma_fst = sma(median_price, period_1)
    sma_snd = sma(median_price, period_2)
    ao = sma_fst - sma_snd
    return ao


def acceleration_deceleraion(p_h, p_l, period_1, period_2, period_3):
    ao = awesome_oscillator(p_h, p_l, period_1, period_2)
    ac = ao - sma(ao, period_3)
    return ac


def smma(p_c, N):
    smma = p_c[N:].ewm(alpha=1.0 / N).mean()
    return smma


def alligator(p_h, p_l, jaw_p, teeth_p, lips_p):
    median_price = (p_h + p_l) / 2
    gator_jaw = smma(median_price, jaw_p)
    gator_teeth = smma(median_price, teeth_p)
    gator_lips = smma(median_price, lips_p)
    return gator_jaw, gator_teeth, gator_lips


def bollinger_bands(p_c, N, D):
    middle_line = sma(p_c, N)
    top_line = middle_line + (D * p_c.std())
    bottom_line = middle_line - (D * p_c.std())
    return top_line, middle_line, bottom_line


def relative_strength_index(p_c, N):
    U, D = [0], [0]
    i = 0
    while i + 1 < len(p_c):
        if p_c[i + 1] > p_c[i]:
            U.append(p_c[i + 1] - p_c[i])
            D.append(0)
            i += 1
        elif p_c[i + 1] < p_c[i]:
            U.append(0)
            D.append(p_c[i] - p_c[i + 1])
            i += 1
        else:
            U.append(0)
            D.append(0)
            i += 1
    U = smma(pd.Series(U, index=p_c.index), N)
    D = smma(pd.Series(D, index=p_c.index), N)
    RS = U / D
    rsi = 100 - (100 / (1 + RS))
    return rsi


def exponential_moving_average(p_c, M):
    weights = np.exp(np.linspace(-1., 0., M))
    weights /= weights.sum()
    a = np.convolve(p_c, weights, mode='full')[:len(p_c)]
    a[:M] = a[M]
    a = pd.Series(a, index=p_c.index)
    return a


def moving_average_convergence(p_c):
    macd = exponential_moving_average(p_c, 12) - exponential_moving_average(p_c, 26)
    signal = exponential_moving_average(macd, 9)
    return macd, signal
