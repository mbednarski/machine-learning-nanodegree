from __future__ import print_function

import itertools
import numpy as np
import gym
import os
import sys

from LearningAgent import LearningAgent
from monitor import Monitor

# sys.stdout = open(os.devnull, 'w')
from plotter import Plotter

max_episodes = 2000

env = gym.envs.make("LunarLander-v2")
mon = Monitor(max_episodes)

agent = LearningAgent(env)

for i_episode in range(max_episodes):
    print('Episode {}/{} epsilon: {}'.format(i_episode, max_episodes, agent.epsilon))
    observation = env.reset()
    state = agent.featurize_observation(observation)

    agent.new_episode(i_episode)

    mon.observe_episode(i_episode, agent.epsilon)

    for t in itertools.count():
        if t >= 10000:
            print('aborting...')
            break

        action = agent.get_action(state)
        next_observation, reward, done, _ = env.step(env.action_space.sample())
        next_state = agent.featurize_observation(next_observation)
        agent.learn(state, next_state, action, reward)
        state = next_state

        print('e: {} t: {} r: {}'.format(i_episode, t, reward))
        mon.observe_step(i_episode, reward)

        if done:
            print('Episode finished')
            break

    mon.save(i_episode, force=False)

mon.save(i_episode, force=True)

p = Plotter()
p.plot_epsilon()
