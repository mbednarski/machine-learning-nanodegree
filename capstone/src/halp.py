from __future__ import print_function, division
import itertools as iter

import random

import numpy as np
import gym
from gym import wrappers

from sklearn.kernel_approximation import RBFSampler

from hdf5monitor import Hdf5Monitor

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, directory='xddd_luna', force=True)

alpha = 0.4
gamma = 0.95
epsilon = 1.
epochs = 1000

monitor = Hdf5Monitor(env, None)
monitor.construct()

n_features = 64
n_actions = env.action_space.n

approximators = np.random.normal(loc=0, scale=1.0, size=(n_actions, n_features))
biases = np.ones(n_actions)

samples = [env.observation_space.sample() for _ in range(50000)]
samples = np.array(samples)

sampler1 = RBFSampler(n_components=n_features // 4, gamma=5.0)
sampler1.fit(samples)
sampler2 = RBFSampler(n_components=n_features // 4, gamma=2.0)
sampler2.fit(samples)
sampler3 = RBFSampler(n_components=n_features // 4, gamma=1.0)
sampler3.fit(samples)
sampler4 = RBFSampler(n_components=n_features // 4, gamma=0.5)
sampler4.fit(samples)


def make_features(observation):
    rbf1 = sampler1.transform([observation])[0]
    rbf2 = sampler2.transform([observation])[0]
    rbf3 = sampler3.transform([observation])[0]
    rbf4 = sampler4.transform([observation])[0]
    return np.concatenate((rbf1, rbf2, rbf3, rbf4))


def approximate_Q(state):
    return np.matmul(approximators, state)


def get_max_action(state):
    q = approximate_Q(state)
    a = np.argmax(q)
    return a


def getV(state):
    q = approximate_Q(state)
    return np.max(q)

for epo in range(epochs):
    obs = env.reset()
    state = make_features(obs)
    creward = 0
    for t in iter.count():
        if random.random() < epsilon:
            action = env.action_space.sample()
            print('random')
        else:
            action = get_max_action(state)

        new_obs, reward, done, _ = env.step(action)

        creward += reward

        new_state = make_features(new_obs)

        if done:
            err = reward - approximate_Q(state)[action]
        else:
            err = reward + gamma * getV(new_state) - approximate_Q(state)[action]

        delta = alpha * err * state

        approximators[action, :] += delta

        state = new_state

        if done:
            print('Epo {} Finished after {} steps. Eps {} alpha {} creward \t{}'.format(epo, t, epsilon, alpha, creward))
            monitor.append('crewards', creward)
            monitor.append('epsilons', epsilon)
            monitor.append('episode_lens', t)
            monitor.append('alphas', alpha)


            epsilon *= 0.99
            alpha *= 0.99
            break

env.close()