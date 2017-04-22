import gym

import numpy as np
import matplotlib as plt
import logging
import matplotlib.pyplot as plt
import itertools
import time


class NeuralNetPolicy(object):
    def __init__(self, n_features, n_actions, n_hidden1=32, n_hidden2=32):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

    def unpack(self, theta):
        shapes = [
            (self.n_features, self.n_hidden1),
            (1, self.n_hidden1),
            (self.n_hidden1, self.n_hidden2),
            (1, self.n_hidden2),
            (self.n_hidden2, self.n_actions),
            (1, self.n_actions),
        ]
        result = []
        start = 0
        for i, offset in enumerate(np.prod(shape) for shape in shapes):
            result.append(theta[start:start + offset].reshape(shapes[i]))
            start += offset
        return result

    def choose_action(self, values):
        return np.argmax(values)

    def forward_pass(self, theta, state):
        w, b, w2, b2, w3, b3 = self.unpack(theta)

        z = state.dot(w) + b
        a1 = np.tanh(z)

        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)

        z3 = a2.dot(w3) + b3
        a3 = np.tanh(z3)

        action = self.choose_action(a3)

        return action

    def get_number_of_parameters(self):
        return (self.n_features + 1) * self.n_hidden1 + (self.n_hidden1 + 1) * self.n_hidden2 + (self.n_hidden2 + 1) * self.n_actions


class PGPEAgent(object):
    def __init__(self, env, alpha_sigma=0.00001, alpha_u=0.0001, initial_sigma=0.5, history_size=50, population_size=500,
                 test_iterations=200):

        self.test_iterations = test_iterations
        self.population_size = population_size
        self.history_size = history_size
        self.alpha_u = alpha_u
        self.alpha_sigma = alpha_sigma
        self.n_features = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.validation_env = gym.make(env.spec.id)
        self.validation_env = gym.wrappers.Monitor(self.validation_env, directory='/tmp/pgpe', force=True)

        self.policy = NeuralNetPolicy(self.n_features, self.n_actions)
        self.P = self.policy.get_number_of_parameters()
        self.N = population_size

        self.T = np.zeros((self.P, self.N))
        self.S = np.zeros((self.P, self.N))

        self.initial_sigma = initial_sigma

        self.b = 0.0

    def evaluate_policy(self, theta, env=None, render=False):
        if env is None:
            env = gym.make(self.env.spec.id)
        episode_reward = 0
        state = env.reset()

        for _ in itertools.count():
            action = self.policy.forward_pass(theta, state)

            new_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            state = new_state
            episode_reward += reward

            if done:
                return episode_reward


    def run(self, max_episodes=150000):
        r_history = []
        val_history = []
        mean_history = []
        cross_history = []
        test_phase = False
        u = np.repeat(0.0, self.P)
        sigma = np.repeat(self.initial_sigma, self.P)

        for _ in range(max_episodes):
            if test_phase:
                val = self.evaluate_policy(u, self.validation_env, render=True)
                self.test_iterations -= 1
                print('Test evaluation with score {}'.format(val))
                if self.test_iterations == 0:
                    break
                continue

            theta = np.zeros((self.N, self.P))
            r = np.zeros(self.N)
            for n in range(self.N):
                theta[n, :] = np.random.normal(u, self.sigma)
                r[n] = self.evaluate_policy(theta[n])


            T = theta.T - np.repeat(np.array([u]).T, self.N, axis=1)

            S = np.divide(np.square(T.T) - np.square(sigma),
                          sigma).T

            mean_history.append(np.mean(r))

            r = r - b
            r = r.T

            val_score = self.evaluate_policy(u, self.validation_env, render=True)
            r_history.append(val_score)

            b = np.mean(r_history[-self.history_size:])
            b_200 = np.mean(r_history[-20:])

            if b_200 > 200:
                test_phase = True
                continue

            u += self.alpha_u * np.matmul(T, r)
            sigma += self.alpha_sigma * np.matmul(S, r)

            val_history.append(val_score)
            cross_history.append(np.mean(val_history[-100:]))
            print(np.mean(val_score))
            plt.clf()
            plt.plot(val_history)
            plt.plot(mean_history)
            plt.plot(cross_history)
            plt.legend(['validation', 'mean', 'benchmark'])
            plt.savefig('xd.png')
            plt.pause(0.05)


def launch():
    logging.disable(logging.CRITICAL)  # disable messagesa about env creation
    np.seterr('raise')  # raise error on arithmetic errors (overflow/underflow, illegal division etc) istead of warnings
    problem = 'LunarLander-v2'
    env = gym.make(problem)

    agent = PGPEAgent(env, alpha_sigma=0.00001, alpha_u=0.0001, history_size=50, population_size=500,
                      test_iterations=200)


if __name__ == '__main__':
    launch()




