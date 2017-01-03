from  __future__ import print_function, division
import numpy as np

class LearningAgent:
    def __init__(self, env):
        self.env = env
        self.epsilon = self.decay(0)

    def featurize_observation(self, observation):
        return np.array(observation)

    def get_action(self, state):
        return None

    def learn(self, state, next_state, action, reward):
        pass

    def new_episode(self, i_episode):
        self.epsilon = self.decay(i_episode)

    def decay(self, t):
        return 1. - 0.0005 * t