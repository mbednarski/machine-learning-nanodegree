import numpy as np

import skopt
import gym
from skopt.space import Dimension, Integer

from knn_agent import KNNSARSAAgent
from skopt.plots import plot_convergence

np.random.seed(1)


def f(x):
    alpha = x[0]
    gamma = x[1]
    density = int(x[2])
    lambda_ = x[3]
    k = x[4]
    t = x[5]
    initial_q = x[6]
    t_decay = x[7]

    env = gym.envs.make("CartPole-v1")
    # env = gym.wrappers.Monitor(env, '/tmp/cartpolev1-ex-2', force=True)
    p = KNNSARSAAgent(env,
                      # np.array([-4.0, -3.2742672, -0.42, -3.60042572]),
                      # np.array([4.0, 3.21788216, 0.42, 3.67279315]),
                      np.array([2.95593077, -3.2742672, -0.24515077, -3.60042572]),
                      np.array([2.4395082, 3.21788216, 0.2585552, 3.67279315]),
                      # np.array([-1.0, -2.0, -1.0, -2.0]),
                      # np.array([1.0, 2.0, 1.0, 2.0]),
                      max_episodes=2500
                      )
    p.set_parameters(**{
        'initial_Q': initial_q,
        'gamma': gamma,
        'alpha': alpha,
        'density': density,
        'lambda': lambda_,
        'k': k,
        't': t,
        't_decay': t_decay

    })
    p.initialize()
    score = p.run()
    env.close()
    return -score


res = skopt.gp_minimize(f,
                        [(0.0, 1.0),
                         (0.0, 1.0),
                         Integer(1, 20),
                         (0.0, 1.0),
                         Integer(1, 16),
                         (5.0, 100.0),
                         (0.0, 100.0),
                         (0.0, 1.0)],
                        n_jobs=2,
                        n_calls=120,
                        verbose=True
                        )
skopt.dump(res, 'opires')
plot_convergence(res)
print(res)
