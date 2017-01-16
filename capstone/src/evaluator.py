from __future__ import print_function

import itertools
import numpy as np
import gym
import os
import sys

from learning_agent import LearningAgent
from monitor import Monitor

# sys.stdout = open(os.devnull, 'w')
from plotter import Plotter

max_episodes = 20000

env = gym.envs.make("LunarLander-v2")
mon = Monitor(max_episodes)
env.reset()

agent = LearningAgent(env)

for i_episode in range(max_episodes):
    # print('Episode {}/{} epsilon: {}'.format(i_episode, max_episodes, agent.epsilon))
    observation = env.reset()
    # state = agent.featurize_observation(observation)

    agent.new_episode(i_episode)

    mon.observe_episode(i_episode, agent.epsilon)

    for t in itertools.count():
        if t >= 10000:
            print('aborting...')
            break

        action = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        agent.learn(observation, next_observation, action, reward)

        mon.observe_step(i_episode, observation, next_observation, action, reward)

        observation = next_observation

        if i_episode % 75 == 0:
            env.render()

        # print('e: {} t: {} r: {} e:{}'.format(i_episode, t, reward, agent.epsilon))

        if done:
            print('Episode finished')
            print('e: {} t: {} r: {} e:{} a:{}'.format(i_episode, t, reward, agent.epsilon, agent.alpha))
            break


# mon.save(i_episode)

p = Plotter()
p.plot_epsilon()