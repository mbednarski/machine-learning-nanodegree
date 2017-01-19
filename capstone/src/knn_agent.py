from __future__ import print_function, division

import random

import numpy as np
import gym
import time
from sklearn.neighbors import NearestNeighbors

import logging


from hdf5monitor import Hdf5Monitor


class BaseAgent(object):
    def __init__(self, env, name):
        self.observation_size = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.name = name
        self.env = env
        self.parameters = dict()
        self.i_episode = 1
        self.monitor = Hdf5Monitor(env, self)

    def store_episode_stats(self, creward, epsilon, duration, alpha):
        self.monitor.append_creward(creward)
        self.monitor.append_epsilon(epsilon)
        self.monitor.append_episode_len(len(self.monitor.epsilons))
        # self.monitor.append_duration(duration)
        # self.monitor.append_epsilon(epsilon)

    def store_step_stats(self, observation, state):
        self.monitor.append_observation(observation)
        self.monitor.append_state(state)

    def get_parameters(self):
        return self.parameters


class KNNSARSAAgent(BaseAgent):
    def __init__(self, env, min_obs, max_obs, max_episodes=100, max_steps=1000 ):
        super(KNNSARSAAgent, self).__init__(env, 'kNN SARSA')

        self.min_obs = min_obs
        self.max_obs = max_obs
        self.n_features = self.env.observation_space.shape[0]
        self.max_steps = max_steps
        self.max_episodes = max_episodes

        self.parameters = self.get_default_parameters()

    def initialize(self):
        logging.warn('Initializing {} with parameters {}'.format(self.name, self.get_parameters()))
        start = time.time()
        self.statelist = self.create_classifiers()
        self.nn = NearestNeighbors(n_neighbors=self.parameters['k'])
        self.nn.fit(self.statelist)
        self.monitor.construct()
        duration = time.time() - start
        logging.warn('Initialized in {:4.2f} seconds'.format(duration))

    def run(self):
        self.game()

    def get_state_size(self):
        return self.n_features

    def get_default_parameters(self):
        return {'alpha': 0.3, 'gamma': 0.9, 'k': 4, 'density': 15, 'lambda': 0.95, 'epsilon': 0.0, 'initial_Q': 10.0}

    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def create_classifiers(self):
        step = 2.0 / self.parameters['density']
        frm = -1.
        to = 1.001
        xy = np.mgrid[frm:to:step, frm:to:step, frm:to:step, frm:to:step].reshape(self.n_features, -1).T
        return xy

    def getkNNSet(self, state):
        d, knn = self.nn.kneighbors([state])
        d = d[0]
        knn = knn[0]

        p = np.divide(1.0, (1.0 + np.multiply(d, d)))
        p = np.divide(p, np.sum(p))
        return knn, p

    def getValues(self, Q, knn, p):
        V = Q[knn, :].T.dot(p)
        return V

    def getBestAction(self, V):
        a = np.argmax(V)
        return a

    def e_greedy_selection(self, V):
        action_size = V.shape[0]
        if random.random() > self.parameters['epsilon']:
            a = self.getBestAction(V)
        else:
            a = random.choice(range(action_size))
        return a

    def update(self, Q, V, V2, knn, p, r, a, ap, trace, done):
        trace[knn, :] = 0.0
        trace[knn, a] = p

        if done:
            delta = r - V[a]
        else:
            delta = (r + np.multiply(self.parameters['gamma'], V2[ap])) - V[a]

        Q = Q + self.parameters['alpha'] * np.multiply(delta, trace)
        trace = self.parameters['gamma'] * np.multiply(self.parameters['lambda'], trace)
        return Q, trace

    def normalize_state(self, state):
        return 2 * ((state - self.min_obs) / (self.max_obs - self.min_obs)) - 1

    def episode(self, Q, trace ):
        state = self.normalize_state(self.env.reset())
        total_reward = 0.
        steps = 0
        knn, p = self.getkNNSet(state)
        V = self.getValues(Q, knn, p)
        a = self.e_greedy_selection(V)

        for i in range(self.max_steps):

            # if not i % 100:
            #     print('.')

            action = a

            next_observation, reward, done, _ = self.env.step(action)
            next_state = self.normalize_state(next_observation)
            # print('{:5}'.format(next_observation))

            total_reward += reward

            knn2, p2 = self.getkNNSet(next_state)
            V2 = self.getValues(Q, knn2, p2)
            ap = self.e_greedy_selection(V2)

            Q, trace = self.update(Q, V, V2, knn, p, reward, a, ap, trace, done)

            a = ap
            state = next_state
            knn = knn2
            p = p2
            V = V2

            steps += 1

            self.store_step_stats(next_observation, state)

            # self.env.render()
            if done:
                # print(total_reward, self.parameters['epsilon'], self.i_episode)
                break

        return total_reward, steps, Q, trace

    def game(self):
        n_states = self.statelist.shape[0]
        Q = np.ones((n_states, self.n_actions)) * self.parameters['initial_Q']
        trace = np.zeros((n_states, self.n_actions))
        epsilon = self.parameters['epsilon']

        for i in range(self.max_episodes):
            # episode
            self.i_episode = i

            start = time.time()
            total_reward, steps, Q, trace = self.episode(Q, trace)
            duration = time.time() - start
            self.store_episode_stats(total_reward, epsilon, duration, self.parameters['alpha'])

            logging.warn('Episode {:4d} finished in {:4.2f} ({:6.6f} sec./step) with score {}'.format(i, duration, duration / steps, total_reward))

            trace.fill(0)
            self.parameters['epsilon'] *= 0.9


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = gym.envs.make("CartPole-v1")
    p = KNNSARSAAgent(env,
                      np.array([-1.0, -2.0, -1.0, -2.0]),
                      np.array([1.0, 2.0, 1.0, 2.0]))
    p.set_parameters(**{
        'density': 50,
        'lambda': 0.8,
        'k': 10
    })
    p.initialize()
    p.run()
