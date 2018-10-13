# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

import keras
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from utils import prep_batch, prep_input


class NN:
    def __init__(self, n_states, n_actions, batch_size, size_hidden,
                 learning_rate, activation):
        self.learning_rate = learning_rate
        self.act = activation
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = self._make_model(n_states, n_actions, size_hidden)
        self.model_t = self._make_model(n_states, n_actions, size_hidden)
        self.batch_size = batch_size

    def _make_model(self, n_states, n_actions, size_hidden):
        model = Sequential()
        model.add(Dense(size_hidden, input_dim=4, activation=self.act))
        model.add(Dense(size_hidden, activation=self.act))
        model.add(Dense(n_actions, activation='linear'))
        opt = SGD(lr=self.learning_rate, momentum=0.5, decay=1e-6, clipnorm=2)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        loss = self.model.fit(X,
                              y,
                              batch_size=self.batch_size,
                              nb_epoch=1,
                              verbose=0,
                              shuffle=True)

        return loss

    def predict(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        if usetarget:
            q_vals = self.model_t.predict(state)
        else:
            q_vals = self.model.predict(state)
        return q_vals[0]

    def update_target(self):
        weights = self.model.get_weights()
        self.model_t.set_weights(weights)
        self.save('weights.h5')
        pass

    def best_action(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        q_vals = self.predict(state, usetarget)
        best_action = np.argmax(q_vals)
        return best_action

    def save(self, fname):
        self.model.save_weights(fname, overwrite=True)
        pass

    def load(self, fname):
        self.model.load_weights(fname)
        self.update()
        pass
