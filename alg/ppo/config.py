import os
import torch
import sys
from network import PolicyNetwork, ValueNetwork
import numpy as np
from datetime import datetime
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from base_policy import base_policy
from env.gym_env import RoadCharging

class Config:
    def __init__(self):
        # This data is only used to identify the problem, such as the number of chargers and EVs, etc.
        self.demo_data_file = "data/all_days_negativePrices_highInitSoC_1for5/config1_5EVs_1chargers.json"
        env = RoadCharging(self.demo_data_file)
        self.charger_num = env.m
        self.ev_num = env.n
        self.c_rates = env.c_rates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork
        self.value_net = ValueNetwork
        self.state_length = len(self.state_shaping(env.reset()))
        self.charging_threshold = 0.5

        self.rl_train_episodes = 20000
        self.policy_lr = 1e-6
        self.value_lr = 1e-6
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.exploration_epsilon = 0.2
        self.buffer_size = 500
        self.rl_test_frequency = 100

        self.bc_num_samples = 100000
        self.bc_lr = 1e-5
        self.bc_train_episodes = 100
        self.bc_batch_size = 64
        self.bc_test_frequency = 10

        self.penalty = 0
        self.init_expert_prob = 0.01
        self.expert_prob_decay = 0.98
        self.expert_policy = base_policy

        self.test_dir = "data/all_days_negativePrices_highInitSoC_1for5"

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", f"exp_{date_str}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_config()

    def state_shaping(self, state: dict) -> np.array:
        time_step = [state['TimeStep'][0] / 96]
        ride_time = state['RideTime']
        charging_status = state['ChargingStatus']
        soc = state['SoC']
        flat_state = np.concatenate([time_step, ride_time, charging_status, soc])
        return flat_state

    def action_mask(self, state: dict) -> list:
        mask = [0] * self.ev_num
        for i in range(self.ev_num):
            if state["RideTime"][i] < 1 and state["SoC"][i] <= 1 - self.c_rates[i]:
                mask[i] = 1
        return mask

    def log_config(self):
        config_file_path = os.path.join(self.output_dir, 'config.txt')
        with open(config_file_path, 'w') as f:
            for attribute, value in self.__dict__.items():
                f.write(f"{attribute}: {value}\n")
        print(f"Config logged to {config_file_path}")