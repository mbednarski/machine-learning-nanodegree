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
        epsilon_file = './monitor/{}/last_epsilon.npy'.format(newest)
        episode_len_file = './monitor/{}/last_episode_len.npy'.format(newest)
        creward_file = './monitor/{}/last_creward.npy'.format(newest)
        self.epsilon_data =np.load(epsilon_file)
        self.episode_len_data =np.load(episode_len_file)
        self.creward_data =np.load(creward_file)

    def plot_epsilon(self):
        plt.plot(self.epsilon_data)
        plt.show()

    def plot_episode_len(self):
        plt.plot(self.episode_len_data)
        plt.show()

    def plot_creward(self):
        plt.plot(self.creward_data)
        plt.show()

if __name__ == '__main__':
    p = Plotter()
    p.plot_episode_len()
    p.plot_epsilon()
    p.plot_creward()