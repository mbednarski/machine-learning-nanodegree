from __future__ import print_function, division

import random

import numpy as np
import gym
from gym import wrappers

from sklearn.neighbors import NearestNeighbors


# minX = -np.ones(8) * 5
# maxX = np.ones(8) * 5

minX = np.array([-0.88490855, -0.2316932,  -1.25727043, -1.90462545, -3.18837976, -4.55707092,
        ])
maxX = np.array([1.01019154,  1.00227976,  1.52617035,  0.29364061,  2.51579046,  5.63901863,
        ])

minX2 = -np.ones(8) * 0.0
maxX2 = np.ones(8) * 0.0

n_actions = 4
n_features = 6
maxsteps = 5000
maxepisodes = 1300
density = 10

nn = NearestNeighbors(n_neighbors=4)


def create_clasifiers():
    step = 2 / density
    frm = -1.
    to = 1.001
    xy = np.mgrid[frm:to:step, frm:to:step, frm:to:step, frm:to:step, frm:to:step, frm:to:step].reshape(n_features, -1).T
    return xy


env = gym.envs.make("LunarLander-v2")
# env = wrappers.Monitor('/mn/cartpole-experiment-1')(env2)


def getkNNSet(state, statelist, k):
    d, knn = nn.kneighbors([state])
    d = d[0]
    knn = knn[0]

    p = np.divide(1.0, (1.0 + np.multiply(d, d)))
    p = np.divide(p, np.sum(p))
    return knn, p


def getValues(Q, knn, p):
    V = Q[knn, :].T.dot(p)
    return V


def getBestAction(V):
    a = np.argmax(V)
    return a


def e_greedy_selection(V, epsilion):
    action_size = V.shape[0]
    if random.random() > epsilion:
        a = getBestAction(V)
    else:
        a = random.choice(range(action_size))
    return a

def update(Q , V , V2, knn , p , r , a, ap , trace, alpha, gamma, lambda_, done):
    trace[knn,:] = 0.0
    trace[knn, a] = p

    if done:
        delta = r - V[a]
    else:
        delta = (r + np.multiply(gamma, V2[ap])) - V[a]


    Q = Q + alpha * np.multiply(delta, trace)
    trace = gamma * np.multiply(lambda_,  trace)
    return Q, trace

def normalize_state(state):
    return 2 * ((state[:6] - minX) / (maxX - minX)) - 1

epo = 1

def episode(maxsteps, Q, trace, alpha, gamma, epsilon, lambda_, statelist, actionlist, k, ):
    state = normalize_state(env.reset())
    total_reward = 0.
    steps = 0
    knn, p = getkNNSet(state, statelist, k)
    V = getValues(Q, knn, p)
    a = e_greedy_selection(V, epsilon)

    for i in range(maxsteps):
        action = a

        next_observation, reward, done, _ = env.step(action)
        # global minX2
        # global maxX2
        # minima = np.vstack((minX2, next_observation))
        # minX2 = np.min(minima, axis=0)
        # maxima = np.vstack((maxX2, next_observation))
        # maxX2 = np.max(maxima, axis=0)
        #
        # print('--------------')
        # print(minX2)
        # print(maxX2)

        next_state = normalize_state(next_observation)
        # print('{:5}'.format(next_observation))

        total_reward += reward

        knn2, p2 = getkNNSet(next_state, statelist, k)
        V2 = getValues(Q, knn2, p2)
        ap = e_greedy_selection(V2, epsilon)

        Q, trace = update(Q, V, V2, knn, p, reward, a, ap, trace, alpha, gamma, lambda_, done)

        a = ap
        state = next_state
        knn = knn2
        p = p2
        V = V2

        steps = steps + 1
        # print('.')

        env.render()
        if done:
            global epo
            epo += 1
            print(total_reward, epsilon, epo)
            break

    return total_reward, steps, Q, trace


def game(maxepisodes):
    actionlist = range(n_actions)
    statelist = create_clasifiers()

    nn.fit(statelist)
    n_states = statelist.shape[0]
    Q = np.ones((n_states, n_actions))*10
    trace = np.zeros((n_states, n_actions))
    alpha = 0.3
    gamma = 0.999
    lambda_ = 0.95
    epsilon = 1.00
    k = 4
    history = np.zeros(maxepisodes)

    for i in range(maxepisodes):
        # episode
        total_reward, steps, Q, trace = episode(maxsteps, Q, trace, alpha, gamma, epsilon, lambda_, statelist,
                                                actionlist, k)
        history[i] = total_reward


        print('mean10 {}'.format(np.mean(history[i-10:i+1])))
        print('mean50 {}'.format(np.mean(history[i-50:i+1])))

        trace.fill(0)
        epsilon = epsilon * 0.9

game(1300)
env.close()