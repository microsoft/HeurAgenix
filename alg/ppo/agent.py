
import os
import random
import torch
import torch.optim as optim
import numpy as np
from config import Config

class PPOAgent:
    def __init__(self, config: Config):
        self.device = config.device
        self.gamma = config.gamma
        self.clip_epsilon = config.clip_epsilon
        self.exploration_epsilon = config.exploration_epsilon
        self.charger_num = config.charger_num
        self.ev_num = config.ev_num
        self.output_dir = config.output_dir
        self.action_mask = config.action_mask
        self.c_rates = config.c_rates
        self.state_shaping = config.state_shaping
        self.charging_threshold = config.charging_threshold

        self.policy_net = config.policy_net(config.state_length, config.ev_num).to(self.device)
        self.value_net = config.value_net(config.state_length).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.value_lr)

    def select_action_with_exploring(self, state: dict):
        if random.random() <= self.exploration_epsilon:
            action = torch.zeros(self.ev_num, dtype=torch.int64, device=self.device)
            action_mask = self.action_mask(state)
            feasible_indices = [i for i, m in enumerate(action_mask) if m == 1]
            if len(feasible_indices) > 0:
                selected_indices = random.sample(feasible_indices, min(self.charger_num, len(feasible_indices)))
                action[selected_indices] = 1
            log_prob = torch.tensor(0.0, device=self.device)
            return action.cpu().numpy().astype(int), log_prob
        return self.select_action(state)

    def select_action(self, state: dict):
        mask = self.action_mask(state)
        state = torch.tensor(self.state_shaping(state), dtype=torch.float32, device=self.device)
        probs = self.policy_net(state)
        masked_probs = probs * torch.tensor(mask, device=self.device)

        action = torch.zeros_like(probs, dtype=torch.int64, device=self.device)
        log_prob = torch.log(masked_probs[masked_probs > 0]).sum() if masked_probs.sum() > 0 else torch.tensor(0.0, device=self.device)
        indices_above_threshold = torch.where(masked_probs > self.charging_threshold)[0]

        if sum(masked_probs) > 0 and len(indices_above_threshold) > 0:
            sorted_indices = indices_above_threshold[probs[indices_above_threshold].argsort(descending=True)]
            top_k_indices = sorted_indices[:self.charger_num]
            action[top_k_indices] = 1

        return action.cpu().numpy().astype(int), log_prob

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def update(self, trajectories):
        states = torch.tensor([self.state_shaping(t['state']) for t in trajectories], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t['action'] for t in trajectories], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t['reward'] for t in trajectories], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t['done'] for t in trajectories], dtype=torch.float32, device=self.device)
        log_probs = torch.tensor([t['log_prob'] for t in trajectories], dtype=torch.float32, device=self.device)

        # Compute values and next values
        values = self.value_net(states).squeeze(-1)
        next_values = torch.cat((values[1:], torch.tensor([0.0], dtype=torch.float32, device=self.device)))

        # Compute advantages
        advantages = self.compute_advantages(rewards, values, next_values, dones)

        # Compute policy loss
        policy_output = self.policy_net(states)
        dist = torch.distributions.Bernoulli(policy_output)
        new_log_probs = dist.log_prob(actions.float()).sum(dim=1)
        ratios = torch.exp(new_log_probs - log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        value_loss = ((values - (rewards + next_values * (1 - dones))).pow(2)).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss, value_loss

    def save_model(self, model_name: str):
        output_path = os.path.join(self.output_dir, model_name)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict()
        }, output_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])