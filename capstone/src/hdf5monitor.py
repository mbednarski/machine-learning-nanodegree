from __future__ import print_function, division

import datetime
import h5py
import numpy as np
from util import ensure_dir_exists


class Hdf5Monitor(object):
    def __init__(self, env, agent, observation_size):
        self.env = env
        self.agent = agent
        self.now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        # self.now_string = 'xd'
        ensure_dir_exists('monitor')
        self.h5file = h5py.File('monitor/' + self.now_string + '.hdf5', 'w')

        self.observations = self.h5file.create_dataset('observations',
                                                       (1, env.observation_space.shape[0]),
                                                       maxshape=(None, env.observation_space.shape[0]),
                                                       chunks=True,
                                                       dtype='f')
        self.observations_count = 0
        self.observations_size = 0

        self.actions = self.h5file.create_dataset('actions',
                                                  (1,),
                                                  maxshape=(None,),
                                                  chunks=True,
                                                  dtype='i')
        self.actions_count = 0
        self.actions_size = 0

        self.h5file.attrs['created'] = datetime.datetime.now().isoformat()
        self.h5file.attrs['env'] = env.spec.id
        self.h5file.attrs['agent'] = agent.name
        self.h5file.attrs['observation_size'] = env.observation_space.shape[0]
        self.h5file.attrs.update(agent.get_parameters())

        self.h5file.flush()

    def append_observation(self, observation):
        self.observations.resize((self.observations_count + 1, self.env.observation_space.shape[0]))
        self.observations[self.observations_count, :] = observation
        self.observations_count += 1
        if not self.observations_count % 100:
            self.h5file.flush()

    def append_action(self, action):
        self.actions.resize((self.actions_count + 1,))
        self.actions[self.actions_count] = action
        self.actions_count += 1
        if not self.actions_count % 2:
            self.h5file.flush()


if __name__ == '__main__':
    m = Hdf5Monitor(None, None, 1)
    for i in range(100):
        m.new_observation(None)
    with  h5py.File('xd.hdf5', 'r') as hf:
        hf.visit(lambda x: print(x))
        dset = hf['observations']
        print(dset[:, :])
