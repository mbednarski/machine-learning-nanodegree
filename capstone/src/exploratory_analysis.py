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
sns.set(color_codes=True)



names = os.listdir("./monitor")
fnames_dates = map(lambda fname: (fname, datetime.datetime.strptime(fname, '%Y-%m-%d_%H_%M_%S.hdf5')), names)
fnames_dates.sort(key=lambda item: item[1])
latestfname, _ = fnames_dates[-1]
latestfname = 'monitor/' + latestfname

with h5py.File(latestfname, mode='r') as hf:
    for k, v in six.iteritems(hf.attrs):
        print('{}: {}'.format(k,v))


    # actions = hf['actions']
    # sns.distplot(actions)
    # sns.plt.show()
    # sns.distplot()

    observations = hf['observations']
    fig, axes = plt.subplots(ncols=2, nrows=1, squeeze=True)
    axes = axes.T.reshape(2)
    for i, ax in enumerate(axes):
        sns.distplot(observations[:,i], ax=ax)
        ax.set_title('$s_{}$'.format(i))
    # sns.distplot(observations[:,0], ax=ax1)
    # sns.distplot(observations[:,1], ax=ax2)

    plt.show()


    print()





