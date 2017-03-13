from __future__ import print_function, division

from hdf5monitor import Hdf5Monitor


class BaseAgent(object):
    def __init__(self, env, name, max_episodes=100, max_steps=1000, render=False):
        self.observation_size = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.name = name
        self.env = env
        self.parameters = dict()
        self.i_episode = 1
        self.monitor = Hdf5Monitor(env, self)
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.render = render

    def store_episode_stats(self, creward, epsilon, episode_len, episode_duration, alpha):
        self.monitor.append('crewards', creward)
        self.monitor.append('epsilons', epsilon)
        self.monitor.append('episode_lens', episode_len)
        self.monitor.append('episode_durations', episode_duration)
        self.monitor.append('alphas', alpha)

    def store_step_stats(self, observation):
        self.monitor.append('observations', observation)

    def get_parameters(self):
        return self.parameters
