import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np


class NeuralNetAgent:
    def __init__(self, env, epochs):

        self.epsylon = 0.9999
        self.gamma = 0.995
        self.epochs = epochs
        self.env = env
        self.alpha = 0.1
        self.batchSize = 4000
        self.buffer = 6000
        self.replay = []
        self.h = 0
        self.n_featues = 4
        self.n_actions = 2
        model = Sequential()
        model.add(Dense(16, init='lecun_uniform', input_shape=(self.n_featues,)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(8, init='lecun_uniform'))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(self.n_actions, init='lecun_uniform'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self.model = model


    def get_action(self, observation):
        qval = self.model.predict(observation.reshape(1,self.n_featues), batch_size=1)

        if random.random() < self.epsylon:
            action = np.random.randint(0,self.n_actions)
        else:
            action = np.argmax(qval)

        return action

    def learn(self, observation, next_observation, action, reward, done):
        if (len(self.replay) < buffer):  # if buffer not filled, add to it
            self.replay.append((observation, action, reward, next_observation))

        else:  # if buffer full, overwrite old values
            if (self.h < (self.buffer - 1)):
                self.h += 1
            else:
                self.h = 0
            self.replay[self.h] = (observation, action, reward, next_observation)

            minibatch = random.sample(self.replay, self.batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                # Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = self.model.predict(old_state.reshape(1, self.n_featues), batch_size=1)
                newQ = self.model.predict(new_state.reshape(1, self.n_featues), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1, self.n_actions))
                y[:] = old_qval[:]
                if not done:  # non-terminal state
                    update = (reward + (self.gamma * maxQ))
                else:  # terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(self.n_featues, ))
                y_train.append(y.reshape(self.n_actions, ))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # print("Game #: %s" % (i,))
            self.model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=1)

    def new_episode(self, i_episode):
        if self.epsylon > 0.1:
            self.epsylon -= (self.epsylon / self.epochs)

    def featurize_observation(self, observation):
        return None

