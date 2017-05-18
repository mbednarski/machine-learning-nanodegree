import logging
import itertools
from time import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.wrappers

from joblib import Parallel, delayed

from monitor import Monitor

NUMBER_OF_THREADS = 3
problem = 'LunarLander-v2'
logging.disable(logging.CRITICAL)  # disable messages about env creation


class NeuralNetPolicy(object):
    def __init__(self, n_features, n_actions, n_hidden=32):
        """
        Creates a policy that uses forward-pass neural network with two hidden layers
        :param n_features: Number of features of the state (and size of the input layer)
        :param n_actions: Number of actions (and size of the output layer)
        :param n_hidden1: Size of first hidden layer
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden

    @staticmethod
    def unpack(self, theta):
        """
        Unpacks 1D array to separate arrays with weights and biases for neural network in order: w, b, w2, b2, w3, b3
        :param theta: 1D array with all parameters for the policy
        :return: Unpacked parameters
        """
        shapes = [
            (self.n_features, self.n_hidden),
            (1, self.n_hidden),
            (self.n_hidden, self.n_actions),
            (1, self.n_actions),
        ]
        result = []
        start = 0
        for i, offset in enumerate(np.prod(shape) for shape in shapes):
            result.append(theta[start:start + offset].reshape(shapes[i]))
            start += offset
        return result

    def forward_pass(self, theta, state):
        """
        Selects action for policy with given theta for a given state \pi_\theta(s)
        :param theta: Policy parametrization
        :param state: State
        :return: selected action
        """
        w, b, w2, b2 = self.unpack(self, theta)

        z = state.dot(w) + b
        a1 = np.tanh(z)

        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)

        return np.argmax(a2)

    def get_number_of_parameters(self):
        """
        Computes total number of parameters for policy
        :return:
        """
        return (self.n_features + 1) * self.n_hidden + (self.n_hidden + 1) * self.n_actions


class PEPGAgent(object):
    def __init__(self, env, alpha_sigma=0.00001, alpha_u=0.0001, initial_sigma=0.5, history_size=50,
                 population_size=500,
                 test_iterations=200):
        """
        Creates an agent using Parameter-exploring Policy Gradient algorithm
        :param env: OpenAI gym environment to solve. Assuming continuous state space and discrete action space
        :param alpha_sigma: Learning rate for sigma
        :param alpha_u: Learning rate for mu
        :param initial_sigma: Initial sigma
        :param history_size: Size of history for the algorithm to consider
        :param population_size: Size of the population
        :param test_iterations: Number of iterations in test phase
        """
        self.test_iterations = test_iterations
        self.population_size = population_size
        self.history_size = history_size
        self.alpha_u = alpha_u
        self.alpha_sigma = alpha_sigma
        self.n_features = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.validation_env = gym.make(env.spec.id)
        # self.validation_env = gym.wrappers.Monitor(self.validation_env, directory='/tmp/pgpe', force=True)

        self.policy = NeuralNetPolicy(self.n_features, self.n_actions, n_hidden=32)
        self.P = self.policy.get_number_of_parameters()
        self.N = population_size

        self.T = np.zeros((self.P, self.N))
        self.S = np.zeros((self.P, self.N))

        self.initial_sigma = initial_sigma


    def run(self, max_iterations=150000):
        """
        Runs PEPG agent and plots progress in real time.
        :param max_iterations: Limit for performed algorithm iterations
        """
        r_history = []
        val_history = []
        mean_history = []
        rolling_history = []
        test_phase = False
        u = np.repeat(0.0, self.P)
        sigma = np.repeat(self.initial_sigma, self.P)
        b = 0.0

        monitor = Monitor()


        for _ in range(max_iterations):
            if test_phase:
                # in test phase just evaluate policy `test_iteration` times
                evaluation_start_time = time()
                val = evaluate_policy(self.policy, u, self.validation_env, render=True)
                self.test_iterations -= 1
                iteration_duration = time() - evaluation_start_time

                monitor.observe_episode(val, iteration_duration)

                print('Test evaluation with score {}'.format(val))
                if self.test_iterations == 0:
                    break
                continue

            evaluation_start_time = time()

            theta = np.zeros((self.N, self.P))
            # r = np.zeros(self.N)
            for n in range(self.N):
                theta[n, :] = np.random.normal(u, sigma)



                # r[n] = self.evaluate_policy(theta[n])

            # thread_func = lambda x: self.evaluate_policy(x)

            results = Parallel(n_jobs=NUMBER_OF_THREADS)(
                delayed(evaluate_policy)(self.policy, theta[n, :]) for n in range(self.N)
            )

            r = np.array(results)

            T = theta.T - np.repeat(np.array([u]).T, self.N, axis=1)

            S = np.divide(np.square(T.T) - np.square(sigma),
                          sigma).T

            mean_history.append(np.mean(r))

            r = r - b
            r = r.T

            # evaluate current policy

            val_score = evaluate_policy(self.policy, u, self.validation_env, render=False)

            # save weights if we may would like to use pre-trained policy (not implemented)
            np.save('current_policy', u)

            # plot results

            r_history.append(val_score)
            val_history.append(val_score)
            rolling_history.append(np.mean(val_history[-100:]))

            update_plot(val_history, mean_history, rolling_history)

            # compute history

            b = np.mean(r_history[-self.history_size:])
            perfomance_for_test = np.mean(r_history[-20:])

            if perfomance_for_test > 200:
                test_phase = True
                continue

            # update policy

            u += self.alpha_u * np.matmul(T, r)
            sigma += self.alpha_sigma * np.matmul(S, r)

            iteration_duration = time() - evaluation_start_time
            monitor.observe_episode(val_score, iteration_duration)
            print(
                'Iteration completed after {:2f} seconds with validation reward {}'.format(iteration_duration,
                                                                                           val_score))

        monitor.save()

def evaluate_policy(policy, theta, env=None, render=False):
    """
    Evaluates policy with given parametrization theta.
    :param theta: Policy parametrization
    :param env: Env to act inside of. If none, a new one is created
    :param render: Renders world state?
    :return: Cumulative reward for single episode
    """
    if env is None:
        env = gym.make(problem)
    episode_reward = 0
    state = env.reset()

    for _ in itertools.count():
        action = policy.forward_pass(theta, state)

        new_state, reward, done, _ = env.step(action)

        if render:
            env.render()

        state = new_state
        episode_reward += reward

        if done:
            return episode_reward


def update_plot(val_history=None, mean_history=None, rolling_history=None):
    """
    Makes the plot
    :param val_history: History of validation scores
    :param mean_history: History of mean scores for whole population
    :param rolling_history: 100 rolling average of validation scores
    """
    plt.clf()
    plt.plot(val_history)
    plt.plot(mean_history)
    plt.plot(rolling_history)
    plt.legend(['validation', 'mean', '100 average'])
    plt.savefig('frame.png')
    plt.pause(0.05)


def launch():
    np.random.seed(2017)  # set fixed seed in order to get reproducible results
    np.seterr('raise')  # raise error on arithmetic errors (overflow/underflow, illegal division etc
    env = gym.make(problem)

    agent = PEPGAgent(env, alpha_sigma=0.00001, alpha_u=0.0001, history_size=50, population_size=500, initial_sigma=0.4,
                      test_iterations=200)

    agent.run(10000)


if __name__ == '__main__':
    launch()
