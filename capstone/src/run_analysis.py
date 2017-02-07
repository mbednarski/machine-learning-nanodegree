from __future__ import print_function, division

import datetime
import six
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(color_codes=True)

colors = ["dark peach", "amber", "greyish", "faded green", "dusty purple"]
pal = sns.xkcd_palette(colors)
sns.set_palette(pal)

names = os.listdir("./monitor")
fnames_dates = map(lambda fname: (fname, datetime.datetime.strptime(fname, '%Y-%m-%d_%H_%M_%S.hdf5')), names)
fnames_dates.sort(key=lambda item: item[1])
latestfname, _ = fnames_dates[-1]
latestfname = 'monitor/' + latestfname

def make_algorithm_sumamry(fname):
    with h5py.File(fname, mode='r') as hf:
        for k, v in six.iteritems(hf.attrs):
            print('{}: {}'.format(k, v))

        crewards = hf['crewards']

        means = np.zeros(crewards.size)
        means2 = np.zeros(crewards.size)
        for i in range(1, crewards.size):
            means[i] = np.mean(crewards[np.max([i-50,0]):i])
            means2[i] = np.mean(crewards[np.max([i-150,0]):i])


        sns.tsplot(crewards[:,0])
        sns.tsplot(means, color=pal[1])
        sns.tsplot(means2, color=pal[2])
        sns.plt.savefig('algo_res.png', dpi=500)
        sns.plt.show()

        sns.tsplot(hf['epsilons'][:,0])
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
            sns.distplot(observations[:,i], ax=ax)
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

        x1 = pd.Series(observations[:, 0].sample(40000), name='$s_1$')
        x2 = pd.Series(observations[:, 1].sample(40000), name='$s_2$')
        fig = sns.jointplot(x1, x2, kind='kde')

        fig.savefig('exploratory_mountaincar.png', dpi=500)
        # fig.show()
        plt.show()

cartpole_file = r'c:\p\github\machine-learning-nanodegree\capstone\src\monitor\2017-02-07_17_12_51.hdf5'
make_algorithm_sumamry(cartpole_file )
exploratory_cartpole(cartpole_file )

#
# with h5py.File(latestfname, mode='r') as hf:
#     for k, v in six.iteritems(hf.attrs):
#         print('{}: {}'.format(k,v))
#
#
#     # actions = hf['actions']
#     # sns.distplot(actions)
#     # sns.plt.show()
#     # sns.distplot()
#
#     observations = hf['observations']
#
#     # fig, axes = plt.subplots(ncols=2, nrows=1, squeeze=True)
#     # axes = axes.reshape(2)
#     # for i, ax in enumerate(axes):
#     #     sns.distplot(observations[:,i], ax=ax)
#     #     ax.set_title('$s_{}$'.format(i))
#     #
#     # fig.savefig('exploratory_mountaincar.png', dpi=500)
#
#     x1 = pd.Series(observations[:, 0], name='$s_1$')
#     x2 = pd.Series(observations[:, 1], name='$s_2$')
#     fig = sns.jointplot(x1, x2, kind='kde')
#     fig.savefig('exploratory_mountaincar.png', dpi=500)
#
#
#
#
#     plt.show()
#
#     print()





