from __future__ import print_function, division

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six

sns.set(color_codes=True)

colors = ["dark peach", "amber", "greyish", "faded green", "dusty purple"]
pal = sns.xkcd_palette(colors)
sns.set_palette(pal)

cartpole_fname = r'data/cartpole.hdf5'
mountaincar_fname = r'data/mountaincar.hdf5'


def make_algorithm_summary(fname, outfile):
    with h5py.File(fname, mode='r') as hf:
        for k, v in six.iteritems(hf.attrs):
            print('{}: {}'.format(k, v))

        crewards = hf['crewards']

        means = np.zeros(crewards.size)
        means2 = np.zeros(crewards.size)
        for i in range(1, crewards.size):
            means[i] = np.mean(crewards[np.max([i - 50, 0]):i])
            means2[i] = np.mean(crewards[np.max([i - 100, 0]):i])

        sns.tsplot(crewards[:, 0])
        sns.tsplot(means, color=pal[1])
        sns.tsplot(means2, color=pal[2])
        sns.plt.savefig(outfile, dpi=500)
        sns.plt.show()

        sns.tsplot(hf['epsilons'][:, 0])
        sns.plt.title('Epsilon')
        sns.plt.show()


def exploratory_cartpole(fname):
    with h5py.File(fname, mode='r') as hf:
        for k, v in six.iteritems(hf.attrs):
            print('{}: {}'.format(k, v))

        observations = hf['observations']

        fig, axes = plt.subplots(ncols=2, nrows=2, squeeze=True)
        axes = axes.reshape(4)
        for i, ax in enumerate(axes):
            sns.distplot(observations[:, i], ax=ax)
            ax.set_title('$s_{}$'.format(i))
            ax.set_yscale('log')

        print('Minimal values of s: {}'.format(np.min(observations, axis=0)))
        print('maximal values of s: {}'.format(np.max(observations, axis=0)))

        # fig.title('Cartpole exploratory')
        fig.savefig('exploratory_cartpole.png', dpi=500)
        # fig.show()
        plt.show()


def exploratory_mountaincar(fname):
    print('Exploratory analysis for file ' + fname)
    with h5py.File(fname, mode='r') as hf:
        for k, v in six.iteritems(hf.attrs):
            print('{}: {}'.format(k, v))

        observations = hf['observations']
        print('# of observations: {}'.format(observations.shape[0]))

        x1 = pd.Series(observations[:, 0], name='$s_1$')
        x2 = pd.Series(observations[:, 1], name='$s_2$')
        fig = sns.jointplot(x1, x2, kind='kde')

        fig.savefig('exploratory_mountaincar.png', dpi=400)
        plt.show()

if __name__ == '__main__':
    # exploratory_cartpole('data/random_cartpole.hdf5')
    exploratory_mountaincar('data/random_mountaincar.hdf5')