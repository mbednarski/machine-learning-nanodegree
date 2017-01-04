from __future__ import print_function, division

from datetime import datetime

import numpy as np

import util


class Monitor:
    def __init__(self, episodes):
        self.episodes = episodes
        self.epsilon_data = np.zeros(episodes)
        self.creward_data = np.zeros(episodes)
        self.episode_len_data = np.zeros(episodes)
        self.act_data = np.zeros((1,8 + 8 + 1 + 1))
        self.save_frequency = 5
        self.now_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.act_i = 0
        util.ensure_dir_exists('monitor')

    def observe_episode(self, i_episode, epsilon):
        self.epsilon_data[i_episode] = epsilon

    def observe_step(self, i_episode, state, next_state, action, reward):
        self.creward_data[i_episode] += reward
        self.episode_len_data[i_episode] += 1

        if self.act_i >= self.act_data.shape[0]:
            zeros = np.zeros((self.act_data.shape[0]*2,8+8+1+1))
            zeros[:self.act_data.shape[0], :self.act_data.shape[1]] = self.act_data
            self.act_data = zeros
            print('Resized act_data to {}'.format(self.act_data.shape[0]))

        self.act_data[self.act_i][0:8] = state
        self.act_data[self.act_i][8:16] = next_state
        self.act_data[self.act_i][16] = action
        self.act_data[self.act_i][17] = reward
        self.act_i += 1

    def save(self, i_episode, force=False):
        if i_episode % self.save_frequency != 0 and not force:
            return

        util.ensure_dir_exists('monitor/{}'.format(self.now_string))

        epsilon_fname = 'monitor/{}/{}_episode_{:06d}_epsilon.npy'.format(self.now_string, 'xD', i_episode)
        creward_fname = 'monitor/{}/{}_episode_{:06d}_creward.npy'.format(self.now_string, 'xD', i_episode)
        episode_len_fname = 'monitor/{}/{}_episode_{:06d}_episode_len.npy'.format(self.now_string, 'xD', i_episode)
        act_fname = 'monitor/{}/{}_episode_{:06d}_act.npy'.format(self.now_string, 'xD', i_episode)

        np.save(epsilon_fname, self.epsilon_data)
        np.save(creward_fname, self.creward_data)
        np.save(episode_len_fname, self.episode_len_data)
        np.save(act_fname, self.act_data)

        np.save('monitor/{}/last_creward'.format(self.now_string), self.creward_data, allow_pickle=False)
        np.save('monitor/{}/last_epsilon'.format(self.now_string), self.epsilon_data, allow_pickle=False)
        np.save('monitor/{}/last_episode_len'.format(self.now_string), self.episode_len_data, allow_pickle=False)
        np.save('monitor/{}/act'.format(self.now_string), self.act_data, allow_pickle=False)
