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

        fig.savefig('exploratory_cartpole.png', dpi=500)
        # fig.show()
        plt.show()


def exploratory_mountaincar(fname):
    with h5py.File(fname, mode='r') as hf:
        for k, v in six.iteritems(hf.attrs):
            print('{}: {}'.format(k, v))

        observations = hf['observations']

        x1 = pd.Series(observations[:40000, 0], name='$s_1$')
        x2 = pd.Series(observations[:40000, 1], name='$s_2$')
        fig = sns.jointplot(x1, x2, kind='kde')

        fig.savefig('exploratory_mountaincar.png', dpi=500)
        # fig.show()
        plt.show()

fname_exp_cartpole = 'plot_data/exploratory_cartpole.hdf5'
fname_exp_mountaincar = 'plot_data/exploratory_mountaincar.hdf5'
# exploratory_cartpole(fname_exp_cartpole)
exploratory_mountaincar(fname_exp_mountaincar)

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





