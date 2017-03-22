from __future__ import print_function, division

import gym
import numpy as np
import time

from base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env, max_episodes=100, max_steps=1000):
        super(RandomAgent, self).__init__(env, 'Random agent', max_episodes=max_episodes, max_steps=max_steps)
        self.monitor.construct()

    def get_action(self):
        return self.env.action_space.sample()

    def run(self):
        self.game()

    def game(self):
        for ep in range(self.max_episodes):
            self.episode(ep)

    def episode(self, i_episode):
        obs = self.env.reset()
        creward = 0
        start_time = time.time()
        for t in range(self.max_steps):
            if self.render:
                env.render()
            action = self.get_action()
            new_obs, reward, done, _ = self.env.step(action)
            creward += reward
            self.store_step_stats(new_obs)

            if done:
                duration = time.time() - start_time
                print('Episode {} finished with creward {} after {} steps ({} seconds)'.format(i_episode, creward,t, duration))
                self.store_episode_stats(creward, 0.0, t, duration, 0.0)
                return
        duration = time.time() - start_time
        print('Episode {} aborted after {} steps ({} seconds)'.format(i_episode, t, duration))
        self.store_episode_stats(creward, 0.0, t, duration, 0.0)


if __name__ == '__main__':
    env = gym.envs.make("CartPole-v1")
    agent = RandomAgent(env, max_episodes=100, max_steps=10000)
    agent.run()
