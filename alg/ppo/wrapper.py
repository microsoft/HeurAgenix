import os
import numpy as np
import gym
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from env.gym_env import RoadCharging, ConstrainAction

class RoadChargingWrapper(gym.Wrapper):
    def __init__(self, config_fname, mode="train", penalty=-100):
        self.mode = mode
        self.penalty = penalty
        env = RoadCharging(config_fname=config_fname)
        env.stoch_step = True

        super().__init__(env)

    def normalize_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) > 100:
            self.rewards.pop(0)
        mean = np.mean(self.rewards)
        std = np.std(self.rewards) if np.std(self.rewards) > 0 else 1
        return (reward - mean) / std

    def step(self, action):
        try:
            state, reward, done, info = self.env.step(action)
        except AssertionError as e:
            state = self.env.obs
            reward = self.penalty
            done = False
            info = {'error': str(e)}

        return state, reward, done, info