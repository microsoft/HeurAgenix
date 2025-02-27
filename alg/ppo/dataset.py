import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import Config
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from wrapper import RoadCharging

class ExpertDataset(Dataset):
    def __init__(self, config: Config):
        self.states = []
        self.actions = []
        self.demo_data_file = config.demo_data_file
        self.expert_policy = config.expert_policy
        self.bc_num_samples = config.bc_num_samples
        self.state_shaping = config.state_shaping
        self.output_dir = config.output_dir

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def collect_expert_data(self):
        print(f"Start collect data by {self.expert_policy.__name__}")
        env = RoadCharging(self.demo_data_file)
        env.stoch_step = True
        state = env.reset(stoch_step=True)
        for _ in range(self.bc_num_samples):
            action = self.expert_policy(env, state)
            self.states.append(self.state_shaping(state))
            self.actions.append(action)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset(stoch_step=True)
                collected_data = len(self.actions)
        print(f"{collected_data} data collected")
        self.save_data("expert_data.pkl")

    def save_data(self, data_name: str):
        output_path = os.path.join(self.output_dir, data_name)
        print(f"Save expert data to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump({'states': self.states, 'actions': self.actions}, f)
    
    def load_data(self, data_path: str):
        print(f"Load expert data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.states = data['states']
            self.actions = data['actions']