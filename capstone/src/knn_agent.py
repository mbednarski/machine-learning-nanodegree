from __future__ import print_function, division

import random

import numpy as np
import gym
from sklearn.neighbors import NearestNeighbors

from hdf5monitor import Hdf5Monitor


class BaseAgent(object):
    def __init__(self, env, name):
        self.observation_size = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.name = name
        self.env = env

        self.monitor = Hdf5Monitor(env, self)

    def store_episode_stats(self, i_episode, creward, epsilon):
        self.monitor.append_creward(creward)
        self.monitor.append_epsilon(epsilon)
        self.monitor.append_episode_len(len(self.monitor.epsilons))

    def store_step_stats(self, observation, state):
        self.monitor.append_observation(observation)
        self.monitor.append_state(state)

    def get_parameters(self):
        return dict()


class KNNSARSAAgent(BaseAgent):
    def __init__(self, env):
        super(KNNSARSAAgent, self).__init__(env, 'kNN SARSA')
        self.maxX = np.array([1.0, 2.0, 1.0, 2.0])
        self.minX = np.array([-1.0, -2.0, -1.0, -2.0])

        self.n_actions = 2
        self.n_features = 4
        self.maxsteps = 1000
        self.maxepisodes = 100

        self.parameters = self.get_default_parameters()

        self.epo = 1

        self.statelist = self.create_clasifiers(self.parameters['density'])
        self.nn = NearestNeighbors(n_neighbors=self.parameters['k'])
        self.nn.fit(self.statelist)

        self.monitor.construct()

    def get_state_size(self):
        return self.n_features

    def get_default_parameters(self):
        return {'alpha': 0.3, 'gamma': 0.9, 'k': 4, 'density': 15, 'lambda': 0.95, 'epsilon': 0.0, 'initial_Q': 10.0}

    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def create_clasifiers(self, density):
        step = 2 / density
        frm = -1.
        to = 1.001
        xy = np.mgrid[frm:to:step, frm:to:step, frm:to:step, frm:to:step].reshape(self.n_features, -1).T
        return xy

    def getkNNSet(self, state, statelist, k):
        # nn = NearestNeighbors(n_neighbors=k)
        # nn.fit(statelist)
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

    def e_greedy_selection(self, V, epsilion):
        action_size = V.shape[0]
        if random.random() > epsilion:
            a = self.getBestAction(V)
        else:
            a = random.choice(range(action_size))
        return a

    def update(self, Q, V, V2, knn, p, r, a, ap, trace, alpha, gamma, lambda_, done):
        trace[knn, :] = 0.0
        trace[knn, a] = p

        if done:
            delta = r - V[a]
        else:
            delta = (r + np.multiply(gamma, V2[ap])) - V[a]

        Q = Q + alpha * np.multiply(delta, trace)
        trace = gamma * np.multiply(lambda_, trace)
        return Q, trace

    def normalize_state(self, state):
        return 2 * ((state - self.minX) / (self.maxX - self.minX)) - 1

    def episode(self, maxsteps, Q, trace, alpha, gamma, epsilon, lambda_, statelist, actionlist, k, ):
        state = self.normalize_state(self.env.reset())
        total_reward = 0.
        steps = 0
        knn, p = self.getkNNSet(state, statelist, k)
        V = self.getValues(Q, knn, p)
        a = self.e_greedy_selection(V, epsilon)

        for i in range(maxsteps):

            # if not i % 100:
            #     print('.')

            action = a

            next_observation, reward, done, _ = self.env.step(action)
            next_state = self.normalize_state(next_observation)
            # print('{:5}'.format(next_observation))

            total_reward += reward

            knn2, p2 = self.getkNNSet(next_state, statelist, k)
            V2 = self.getValues(Q, knn2, p2)
            ap = self.e_greedy_selection(V2, epsilon)

            Q, trace = self.update(Q, V, V2, knn, p, reward, a, ap, trace, alpha, gamma, lambda_, done)

            a = ap
            state = next_state
            knn = knn2
            p = p2
            V = V2

            steps = steps + 1

            self.store_step_stats(next_observation, state)

            # self.env.render()
            if done:
                self.epo += 1
                print(total_reward, epsilon, self.epo)
                break

        return total_reward, steps, Q, trace

    def game(self, maxepisodes):
        actionlist = range(self.n_actions)

        n_states = self.statelist.shape[0]
        Q = np.ones((n_states, self.n_actions)) * self.parameters['initial_Q']
        trace = np.zeros((n_states, self.n_actions))
        alpha = self.parameters['alpha']
        gamma = self.parameters['gamma']
        lambda_ = self.parameters['lambda']
        epsilon = self.parameters['epsilon']
        k = self.parameters['k']

        for i in range(maxepisodes):
            # episode
            total_reward, steps, Q, trace = self.episode(self.maxsteps, Q, trace, alpha, gamma, epsilon, lambda_,
                                                         self.statelist,
                                                         actionlist, k)

            self.store_episode_stats(i, total_reward, epsilon)

            trace.fill(0)
            self.parameters['epsilon'] *= 0.9


if __name__ == '__main__':
    env = gym.envs.make("CartPole-v1")
    p = KNNSARSAAgent(env)
    p.game(50000)
