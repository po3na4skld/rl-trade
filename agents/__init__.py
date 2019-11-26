from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import Model


class Agent:

    @staticmethod
    def _get_model(state_dim, n_actions):
        inputs = Input(state_dim)

        lstm = LSTM(128, return_sequences=True)(inputs)
        dropout = Dropout(0.1)(lstm)

        lstm = LSTM(128, return_sequences=True)(dropout)
        dropout = Dropout(0.1)(lstm)

        lstm = LSTM(128)(dropout)
        dropout = Dropout(0.1)(lstm)

        dense = Dense(512, activation='relu')(dropout)
        dense = Dense(n_actions)(dense)

        return Model(inputs=inputs, outputs=dense)