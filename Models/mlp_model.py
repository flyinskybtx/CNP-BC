import copy
import os.path as osp
from collections import defaultdict

import numpy as np
from ray.rllib.offline import JsonReader

from tensorflow import keras


class MLPModel:
    def __init__(self, state_dims, action_dims, name='mlp_model'):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.name = name

    def build_model(self, hiddens):
        query_x = keras.layers.Input(name='query_x', shape=(self.state_dims + self.action_dims,))
        x = query_x
        for units in hiddens:
            x = keras.layers.Dense(units, activation='tanh')(x)

        target_y = keras.layers.Dense(self.state_dims, activation='linear')(x)
        self.model = keras.models.Model(inputs=query_x, outputs=target_y, name=self.name)

    def train(self, xs, ys, epochs=100, ):
        self.model.compile(
            # run_eagerly=True,
            optimizer=keras.optimizers.Adam(5e-4),
            loss='mse',
        )

        self.model.fit(
            xs, ys, epochs=epochs, verbose=0,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5, mode='auto',
                    restore_best_weights=True),
            ]
        )

    def predict(self, query_x):
        query_x = query_x.reshape(-1, self.state_dims + self.action_dims)
        query = query_x

        target_y = self.model.predict(query)
        return target_y
