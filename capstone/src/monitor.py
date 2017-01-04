from __future__ import print_function, division

from datetime import datetime

import numpy as np

import util


class Monitor:
    def __init__(self, episodes_count):
        self.chunk_size = 50
        self.act_size = 8 + 8 + 1 + 1 # state, next state, action, reward
        self.saved_steps_counter = 0
        self.episodes_count = episodes_count
        self.epsilon_data = np.zeros(self.chunk_size)
        self.creward_data = np.zeros(self.chunk_size)
        self.episode_len_data = np.zeros(self.chunk_size)
        self.act_data = np.zeros((self.chunk_size, 8 + 8 + 1 + 1))
        self.now_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.episodes_filled = 0
        self.steps_filled = 0
        util.ensure_dir_exists('monitor')

    def observe_episode(self, i_episode, epsilon):
        if self.episodes_filled >= self.chunk_size:
            self.save_episodes(i_episode)

        self.epsilon_data[self.episodes_filled] = epsilon
        self.episodes_filled += 1

    def observe_step(self, i_episode, state, next_state, action, reward):
        if self.steps_filled >= self.chunk_size:
            self.save_steps(i_episode)

        self.creward_data[self.steps_filled] += reward
        self.episode_len_data[self.steps_filled] += 1

        self.act_data[self.steps_filled][0:8] = state
        self.act_data[self.steps_filled][8:16] = next_state
        self.act_data[self.steps_filled][16] = action
        self.act_data[self.steps_filled][17] = reward

        self.steps_filled += 1

    def save(self, i_episode):
        self.epsilon_data = self.epsilon_data[:self.episodes_filled]
        self.creward_data= self.creward_data[:self.episodes_filled]
        self.episode_len_data= self.episode_len_data[:self.episodes_filled]

        self.save_episodes(i_episode)

        self.act_data= self.act_data[:self.steps_filled]
        self.save_steps(i_episode)


    def save_episodes(self, i_episode):
        util.ensure_dir_exists('monitor/{}'.format(self.now_string))

        epsilon_fname = 'monitor/{}/{}_episode_{:06d}_epsilon.npy'.format(self.now_string, 'xD', i_episode)
        creward_fname = 'monitor/{}/{}_episode_{:06d}_creward.npy'.format(self.now_string, 'xD', i_episode)
        episode_len_fname = 'monitor/{}/{}_episode_{:06d}_episode_len.npy'.format(self.now_string, 'xD', i_episode)

        np.save(epsilon_fname, self.epsilon_data, allow_pickle=False)
        np.save(creward_fname, self.creward_data, allow_pickle=False)
        np.save(episode_len_fname, self.episode_len_data, allow_pickle=False)

        self.epsilon_data.fill(0)
        self.creward_data.fill(0)
        self.episode_len_data.fill(0)

        self.episodes_filled = 0

    def save_steps(self, i_episode):
        util.ensure_dir_exists('monitor/{}'.format(self.now_string))
        act_fname = 'monitor/{}/{}_episode_{:06d}_{:06d}_act.npy'.format(self.now_string, 'xD', i_episode, self.saved_steps_counter)
        self.saved_steps_counter += 1
        np.save(act_fname, self.act_data, allow_pickle=False)
        self.act_data.fill(0)
        self.steps_filled = 0



