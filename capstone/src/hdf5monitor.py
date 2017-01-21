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

    def construct(self):
        self.observations = self.h5file.create_dataset('observations',
                                                       (1, self.env.observation_space.shape[0]),
                                                       maxshape=(None, self.env.observation_space.shape[0]),
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


        self.crewards = self.h5file.create_dataset('crewards',
                                                  (1,),
                                                  maxshape=(None,),
                                                  chunks=True,
                                                  dtype='f')
        self.crewards_count = 0
        self.crewards_size = 0

        # states

        self.states = self.h5file.create_dataset('states',
                                                  (1,self.agent.get_state_size()),
                                                  maxshape=(None,self.agent.get_state_size()),
                                                  chunks=True,
                                                  dtype='f')
        self.states_count = 0
        self.states_size = 0

        # epsilon

        self.epsilons = self.h5file.create_dataset('epsilons',
                                                  (1,),
                                                  maxshape=(None,),
                                                  chunks=True,
                                                  dtype='f')
        self.epsilon_count = 0
        self.epsilon_size = 0

        # epsiode len

        self.episode_lens = self.h5file.create_dataset('episode_len',
                                                   (1,),
                                                   maxshape=(None,),
                                                   chunks=True,
                                                   dtype='i')
        self.episode_len_count = 0
        self.episode_len_size = 0

        self.h5file.attrs['created'] = datetime.datetime.now().isoformat()
        self.h5file.attrs['env'] = self.env.spec.id
        self.h5file.attrs['agent'] = self.agent.name
        self.h5file.attrs['observation_size'] = self.env.observation_space.shape[0]
        self.h5file.attrs.update(self.agent.get_parameters())

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


    def append_creward(self, creward):
        self.crewards.resize((self.crewards_count + 1,))
        self.crewards[self.crewards_count] = creward
        self.crewards_count += 1
        if not self.crewards_count % 2:
            self.h5file.flush()

    def append_state(self, state):
        self.states.resize((self.states_count + 1, self.agent.get_state_size()))
        self.states[self.states_count] = state
        self.states_count += 1
        if not self.states_count % 2:
            self.h5file.flush()

    def append_epsilon(self, epsilon):
        self.epsilons.resize((self.epsilon_count + 1,))
        self.epsilons[self.epsilon_count] = epsilon
        self.epsilon_count += 1
        if not self.epsilon_count % 2:
            self.h5file.flush()

    def append_episode_len(self, ep_len):
        self.episode_lens.resize((self.episode_len_count + 1,))
        self.episode_lens[self.episode_len_count] = ep_len
        self.episode_len_count += 1
        if not self.episode_len_count % 2:
            self.h5file.flush()



if __name__ == '__main__':
    m = Hdf5Monitor(None, None, 1)
    for i in range(100):
        m.new_observation(None)
    with  h5py.File('xd.hdf5', 'r') as hf:
        hf.visit(lambda x: print(x))
        dset = hf['observations']
        print(dset[:, :])
