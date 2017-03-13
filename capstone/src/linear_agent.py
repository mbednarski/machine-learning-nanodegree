from __future__ import print_function, division
import itertools as iter

import random

import numpy as np
import gym
import time
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.kernel_approximation import RBFSampler

import logging


from hdf5monitor import Hdf5Monitor
from knn_agent import BaseAgent

np.set_printoptions(precision=4, linewidth=220)

env = gym.make('CartPole-v0')

alpha = 0.7
gamma = 0.9
epsilon = 1.
epochs = 2000

n_features = env.observation_space.sample().shape[0]
n_actions = env.action_space.n

approximators = np.random.normal(loc=0, scale=0.1, size=(n_actions, 400 + 0))
approximators = np.ones((n_actions, 400 + 0))
biases = np.ones(n_actions)

samples = [env.observation_space.sample() for _ in range(50000)]
samples = np.array(samples)

sampler1 = RBFSampler(n_components=100, gamma=5.0)
sampler1.fit(samples)
sampler2 = RBFSampler(n_components=100, gamma=2.0)
sampler2.fit(samples)
sampler3 = RBFSampler(n_components=100, gamma=1.0)
sampler3.fit(samples)
sampler4 = RBFSampler(n_components=100, gamma=0.5)
sampler4.fit(samples)

def make_features(observation):
    rbf1 = sampler1.transform([observation])[0]
    rbf2 = sampler2.transform([observation])[0]
    rbf3 = sampler3.transform([observation])[0]
    rbf4 = sampler4.transform([observation])[0]
    return np.concatenate((rbf1, rbf2, rbf3, rbf4))
    # features = np.array(
    #     [
    #         observation[0],
    #         observation[1],
    #         observation[2],
    #         observation[3],
    #         # observation[0] * observation[1],
    #         # observation[0] * observation[2],
    #         # observation[0] * observation[3],
    #         # observation[1] * observation[2],
    #         # observation[1] * observation[3],
    #         # observation[2] * observation[3],
    #     ]
    # )
    # return features

def approximate_Q(state):
    return np.matmul( approximators, state)

def get_max_action(state):
    q = approximate_Q(state)
    a =  np.argmax(q)
    assert a < n_actions
    return a

def getV(state):
    q = approximate_Q(state)
    return np.max(q)

# print (approximate_Q(make_features(np.array([1,2,3,4]))))

for epo in range(epochs):
    obs = env.reset()
    state = make_features(obs)

    for t in iter.count():
        # env.render()

        if random.random() < epsilon:
            action = random.choice(range(n_actions))
        else:
            action = get_max_action(state)

        new_obs, reward, done, _ = env.step(action)

        new_state = make_features(new_obs)

        if done:
            err = reward - approximate_Q(state)[action]
            print('Err {}'.format(err))
        else:
            err = reward + gamma * getV(new_state) - approximate_Q(state)[action]
        # err = reward + gamma * getV(new_state) - approximate_Q(state)[action]


        delta = alpha * err * state

        # for i in range(400):
        #     approximators[action, i] = approximators[action, i] * (1. - alpha)  + delta[i]

        approximators[action, :] += delta

        state = new_state

        if done:
            print('Finished after {} steps. Eps {}'.format(t, epsilon))
            epsilon *= 0.9
            # print(approximators)
            break

