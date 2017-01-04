from __future__ import print_function, division

import datetime
import matplotlib.pyplot as plt
import os
import numpy as np


class Plotter:
    def __init__(self):
        names = os.listdir("./monitor")
        dates = map(lambda name: datetime.datetime.strptime(name, '%Y-%m-%d_%H_%M_%S'), names)
        dates.sort()
        newest = dates[-1]
        newest = newest.strftime('%Y-%m-%d_%H_%M_%S')
        newest = './monitor/' + newest + '/'
        datafiles = os.listdir(newest)

        self.act_data = self.parse_act(datafiles, dirname=newest)
        self.epsilon_data = self.parse_epsilon(datafiles, dirname=newest)
        self.episode_len_data = self.parse_episode_len(datafiles, dirname=newest)
        self.creward_data = self.parse_creward(datafiles, dirname=newest)

    def parse_act(self, names, dirname):
        names = [n for n in names if n.endswith('_act.npy')]
        names_map = map(lambda x: (dirname + x, int(x.split('_')[-2])), names)
        names_map.sort(key=lambda x: x[1])

        files_data = tuple(map(lambda x: np.load(x[0]), names_map))

        arr = np.concatenate(files_data)

        return arr

    def parse_episode_len(self, names, dirname):
        names = [n for n in names if n.endswith('_episode_len.npy')]
        names_map = map(lambda x: (dirname + x, int(x.split('_')[-3])), names)
        names_map.sort(key=lambda x: x[1])

        files_data = tuple(map(lambda x: np.load(x[0]), names_map))

        arr = np.concatenate(files_data)

        return arr

    def parse_creward(self, names, dirname):
        names = [n for n in names if n.endswith('_creward.npy')]
        names_map = map(lambda x: (dirname + x, int(x.split('_')[-2])), names)
        names_map.sort(key=lambda x: x[1])

        files_data = tuple(map(lambda x: np.load(x[0]), names_map))

        arr = np.concatenate(files_data)

        return arr

    def parse_epsilon(self, names, dirname):
        names = [n for n in names if n.endswith('_epsilon.npy')]
        names_map = map(lambda x: (dirname + x, int(x.split('_')[-2])), names)
        names_map.sort(key=lambda x: x[1])

        files_data = tuple(map(lambda x: np.load(x[0]), names_map))

        arr = np.concatenate(files_data)

        return arr

    def plot_epsilon(self):
        plt.plot(self.epsilon_data)
        plt.show()

    def plot_episode_len(self):
        plt.plot(self.episode_len_data)
        plt.show()

    def plot_creward(self):
        plt.plot(self.creward_data)
        plt.show()

    def plot_reward(self):
        plt.plot(self.act_data[:, -1])
        plt.show()


if __name__ == '__main__':
    p = Plotter()
    p.plot_reward()
    p.plot_episode_len()
    p.plot_creward()
    p.plot_epsilon()
