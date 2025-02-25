import os
import torch
from datetime import datetime
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from base_policy import base_policy


class Config:
    def __init__(self):
        # This data is only used to identify the problem, such as the number of chargers and EVs, etc.
        self.data_file = "data/all_days_negativePrices_highInitSoC_1for5/config1_5EVs_1chargers.json"
        self.charger_num = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.penalty = 0
        self.max_train_episodes = 20000
        self.policy_lr = 1e-5
        self.value_lr = 1e-5
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.exploration_epsilon = 0.2

        self.init_expert_prob = 0.5
        self.expert_prob_decay = 0.98
        self.expert_policy = base_policy
        self.max_buffer_size = 500

        self.test_frequency = 100
        self.test_dir = "data/all_days_negativePrices_highInitSoC_1for5"
        

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", f"exp_{date_str}")
        os.makedirs(self.output_dir, exist_ok=True)
