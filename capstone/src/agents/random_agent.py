from __future__ import print_function, division

import gym
import numpy as np

from base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env):
        super(RandomAgent, self).__init__(env, 'Random agent')
        self.monitor.construct()

    def get_action(self):
        return self.env.action_space.sample()

    def run(self):
        self.game()

    def game(self):
        for ep in range(self.max_episodes):
            self.episode()

    def episode(self):
        obs = self.env.reset()
        creward = 0
        for t in range(self.max_steps):
            if self.render:
                env.render()
            action = self.get_action()
            new_obs, reward, done, _ = self.env.step(action)
            creward += reward
            self.store_step_stats(new_obs)

            if done:
                print('Episode finished with creward {}'.format(creward))
                self.store_episode_stats(creward, 0.0, t, 0.0, 0.0)
                break


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")
    agent = RandomAgent(env)
    agent.run()
