from __future__ import print_function, division

import datetime
import weakref

import h5py
import numpy as np
from util import ensure_dir_exists


class Hdf5Monitor(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        # self.now_string = 'xd'
        ensure_dir_exists('monitor')
        self.h5file = h5py.File('monitor/' + self.now_string + '.hdf5', 'w')
        self.collections = dict()

    def add_collection(self, name, size=1, dtype='f'):
        shape = (10, size) if size else (10,)
        maxshape = (None, size) if size else (None,)
        self.collections[name] = self.h5file.create_dataset(name, shape=shape, dtype=dtype, chunks=True,
                                                            maxshape=maxshape)
        self.collections[name].attrs['inserted'] = 0

    def append(self, name, value):
        collection = self.collections[name]
        if collection.attrs['inserted'] >= collection.shape[0]:
            collection.resize((collection.shape[0] + 50, collection.shape[1]))

        collection[collection.attrs['inserted'], :] = value
        collection.attrs['inserted'] += 1

    def construct(self):
        self.add_collection('observations', self.env.observation_space.shape[0])
        self.add_collection('actions', 1, dtype='i')
        self.add_collection('crewards', 1)
        self.add_collection('states', self.agent.get_state_size())
        self.add_collection('epsilons', 1, 'f')
        self.add_collection('alphas', 1, 'f')
        self.add_collection('episode_lens', 1, 'i')
        self.add_collection('episode_durations', 1, 'f')


        self.h5file.attrs['created'] = datetime.datetime.now().isoformat()
        self.h5file.attrs['env'] = self.env.spec.id
        self.h5file.attrs['agent'] = self.agent.name
        self.h5file.attrs['observation_size'] = self.env.observation_space.shape[0]
        self.h5file.attrs.update(self.agent.get_parameters())

        self.h5file.flush()

