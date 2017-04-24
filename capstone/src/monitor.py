from __future__ import print_function, division

from datetime import datetime

import numpy as np

import util


class Monitor:
    def __init__(self):
        """
        Creates a new monitor that will store data in a directory with current timestamp
        """
        self.chunk_size = 10
        self.creward_data = np.zeros(self.chunk_size)
        self.epsilon_data = np.zeros(self.chunk_size)
        self.episode_duration = np.zeros(self.chunk_size)
        self.now_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.episodes_filled = 0
        util.ensure_dir_exists('monitor')
        self.i_episode = 0

    def observe_episode(self, episode_reward, duration):
        """
        Appends data about single episode (reward) and duration of learning phase. Saves to disk if needed
        :param episode_reward: Episode reward
        :param duration: Learning phase duration
        """
        if self.episodes_filled >= self.chunk_size:
            self.save()

        self.creward_data[self.episodes_filled] = episode_reward
        self.episode_duration[self.episodes_filled] = duration
        self.episodes_filled += 1
        self.i_episode += 1

    def save(self):
        """
        Enforces saving on disk
        """
        self.epsilon_data = self.epsilon_data[:self.episodes_filled]
        self.creward_data= self.creward_data[:self.episodes_filled]
        self.episode_duration= self.episode_duration[:self.episodes_filled]

        util.ensure_dir_exists('monitor/{}'.format(self.now_string))

        creward_fname = 'monitor/{}/episode_{:06d}_creward.npy'.format(self.now_string, self.i_episode)
        episode_duration_fname = 'monitor/{}/episode_{:06d}_episode_duration.npy'.format(self.now_string, self.i_episode)

        np.save(creward_fname, self.creward_data, allow_pickle=False)
        np.save(episode_duration_fname, self.episode_duration, allow_pickle=False)

        self.creward_data.fill(0)
        self.episode_duration.fill(0)

        self.episodes_filled = 0





