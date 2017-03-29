from __future__ import print_function, division

import random

import numpy as np
import gym
import time
from sklearn.neighbors import NearestNeighbors

import logging

from base_agent import BaseAgent
from hdf5monitor import Hdf5Monitor



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
        return self.game()

    def get_state_size(self):
        return self.n_features

    def get_default_parameters(self):
        return {'alpha': 0.3, 'gamma': 0.9, 'k': 4, 'density': 15, 'lambda': 0.95, 'epsilon': 0.0, 'initial_Q': 10.0,
                't': 100.0, 't_decay': 0.99}

    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def create_classifiers(self):
        step = 2.0 / self.parameters['density']
        frm = -1.
        to = 1.001
        # xy = np.mgrid[frm:to:step, frm:to:step].reshape(self.n_features, -1).T
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
        probs = np.divide(
            np.exp(np.divide(V,self.parameters['t'] )),
            np.sum(
                np.exp(np.divide(V, self.parameters['t'] ))
            )
        )
        return np.random.choice(range(probs.shape[0]), p=probs)
        #
        # action_size = V.shape[0]
        # if random.random() > self.parameters['epsilon']:
        #     a = self.getBestAction(V)
        # else:
        #     a = random.choice(range(action_size))
        # return a



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

            self.store_step_stats(next_observation)

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

        ep_mean_scores = np.zeros(self.max_episodes)

        for i in range(self.max_episodes):
            # episode
            self.i_episode = i

            start = time.time()
            total_reward, steps, Q, trace = self.episode(Q, trace)
            duration = time.time() - start
            self.store_episode_stats(total_reward, epsilon,i, duration, self.parameters['alpha'],
                                     )
            ep_mean_scores[i] = total_reward

            if i == 499:
                logging.warn('Episode {:4d} finished in {:4.2f} ({:6.6f} sec./step) temp: {} with score {}'.format(i, duration, duration / steps, self.parameters['t'], total_reward))

            trace.fill(0)
            self.parameters['epsilon'] *= 0.9
            self.parameters['t'] *= self.parameters['t_decay']
            if self.parameters['t'] < 0.15:
                self.parameters['t'] = 0.15
        return np.mean(ep_mean_scores[-250:])

if __name__ == '__main__':

    np.random.seed(1)
    # def mountaincar_knn():
    # 	print('Evaluating CartPole with kNNAgent')
    # 	env = gym.envs.make("MountainCar-v0")
    # 	p = KNNSARSAAgent(env,
    # 					  np.array([-1.2, -0.07]),
    # 					  np.array([0.6, 0.07]),
    # 					  max_episodes=2000,
    # 					  max_steps=150000
    # 					  )
    # 	p.set_parameters(**{
    # 		'density': 20,
    # 		'lambda': 0.95,
    # 		'gamma': 0.99,
    # 		'k': 6
    # 	})
    # 	p.initialize()
    # 	p.run()

    # mountaincar_knn()

    logging.basicConfig(level=logging.DEBUG)
    env = gym.envs.make("CartPole-v1")
    env = gym.wrappers.Monitor(env, '/tmp/cartpolev1-ex-2', force=True)
    p = KNNSARSAAgent(env,
                      # np.array([-4.0, -3.2742672, -0.42, -3.60042572]),
                      # np.array([4.0, 3.21788216, 0.42, 3.67279315]),
                      np.array([-0.95593077, -3.2742672, -0.24515077, -2.10042572]),
                      np.array([2.4395082, 3.21788216, 0.2585552, 3.67279315]),
                      # np.array([-1.0, -2.0, -1.0, -2.0]),
                      # np.array([1.0, 2.0, 1.0, 2.0]),
                      max_episodes=1500
                      )
    p.set_parameters(**{
    'initial_Q': 10.0,
        'gamma': 0.99,
        'alpha': 0.33,
        'density': 15,
        'lambda': 0.95,
        'k': 4
    })
    p.initialize()
    p.run()
    env.close()