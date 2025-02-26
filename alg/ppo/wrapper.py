import os
import numpy as np
import gym
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from env.gym_env import RoadCharging, ConstrainAction

class RoadChargingWrapper(gym.Wrapper):
    def __init__(self, config_fname, penalty=-100):
        self.penalty = penalty
        env = RoadCharging(config_fname=config_fname)
        env.stoch_step = True

        self.reward_mean = 0
        self.reward_std = 1
        self.eps = 1e-6
        super().__init__(env)

    def step(self, action):
        try:
            state, reward, done, info = self.env.step(action)
        except AssertionError as e:
            state = self.env.obs
            reward = self.penalty
            done = False
            info = {'error': str(e)}

        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * ((reward - self.reward_mean) ** 2)
        reward = (reward - self.reward_mean) / (self.reward_std + self.eps)

        return state, reward, done, info