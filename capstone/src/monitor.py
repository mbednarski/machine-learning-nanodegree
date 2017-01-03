from __future__ import print_function, division

from datetime import datetime

import numpy as np

import util


class Monitor:
    def __init__(self, episodes):
        self.epsilon_data = np.zeros(episodes)
        self.creward_data = np.zeros(episodes)
        self.episode_len_data = np.zeros(episodes)
        self.save_frequency = 5
        self.now_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        util.ensure_dir_exists('monitor')

    def observe_episode(self, i_episode, epsilon):
        self.epsilon_data[i_episode] = epsilon

    def observe_step(self, i_episode, reward):
        self.creward_data[i_episode] += reward
        self.episode_len_data[i_episode] += 1

    def save(self, i_episode, force=False):
        if i_episode % self.save_frequency != 0 and not force:
            return

        util.ensure_dir_exists('monitor/{}'.format(self.now_string))

        epsilon_fname = 'monitor/{}/{}_episode_{:05d}_epsilon.npy'.format(self.now_string, 'xD', i_episode)
        creward_fname = 'monitor/{}/{}_episode_{:05d}_creward.npy'.format(self.now_string, 'xD', i_episode)
        episode_len_fname = 'monitor/{}/{}_episode_{:05d}_episode_len.npy'.format(self.now_string, 'xD', i_episode)

        np.save(epsilon_fname, self.epsilon_data)
        np.save(creward_fname, self.creward_data)
        np.save(episode_len_fname, self.episode_len_data)

        np.save('monitor/{}/last_creward'.format(self.now_string), self.creward_data, allow_pickle=False)
        np.save('monitor/{}/last_epsilon'.format(self.now_string), self.epsilon_data, allow_pickle=False)
        np.save('monitor/{}/last_episode_len'.format(self.now_string), self.episode_len_data, allow_pickle=False)


