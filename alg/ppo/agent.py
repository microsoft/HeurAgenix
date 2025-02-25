import random
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from network import PolicyNetwork, ValueNetwork
from config import Config

class PPOAgent:
    def __init__(self, env, config: Config):
        self.env = env
        self.device = config.device
        self.gamma = config.gamma
        self.clip_epsilon = config.clip_epsilon
        self.exploration_epsilon = config.exploration_epsilon
        self.charger_num = config.charger_num

        input_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
        output_dim = env.action_space.shape[0]
        self.policy_net = PolicyNetwork(input_dim, output_dim).to(self.device)
        self.value_net = ValueNetwork(input_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.value_lr)

    def state_shaping(self, state: dict):
        # Extract and flatten the state components
        time_step = state['TimeStep']
        ride_time = state['RideTime']
        charging_status = state['ChargingStatus']
        soc = state['SoC']
        flat_state = np.concatenate([time_step, ride_time, charging_status, soc])

        return flat_state

    def action_shaping(self, action, num_agents):
        actions = np.zeros((num_agents), dtype=int)
        actions[action] = 1
        return actions

    def select_action_with_explorer(self, state: dict):
        state = torch.tensor(self.state_shaping(state), dtype=torch.float32).to(self.device)
        probs = self.policy_net(state)
        action = torch.zeros_like(probs, dtype=int)
        indices_above_threshold = torch.where(probs > 0.5)[0]
        
        if len(indices_above_threshold) > 0:
            sorted_indices = indices_above_threshold[probs[indices_above_threshold].argsort(descending=True)]
            top_k_indices = sorted_indices[:self.charger_num]
            action[top_k_indices] = 1
        log_prob = torch.log(probs).sum()

        # ε-greedy exploration
        if random.random() < self.exploration_epsilon:
            action = torch.tensor(np.random.choice([0, 1], size=action.shape), dtype=torch.float32)
            log_prob = torch.tensor(0.0)

        return action.numpy().astype(int), log_prob

    def select_action(self, state: dict):
        state = torch.tensor(self.state_shaping(state), dtype=torch.float32).to(self.device)
        probs = self.policy_net(state)
        action = torch.zeros_like(probs, dtype=int)
        indices_above_threshold = torch.where(probs > 0.5)[0]
        
        if len(indices_above_threshold) > 0:
            sorted_indices = indices_above_threshold[probs[indices_above_threshold].argsort(descending=True)]
            top_k_indices = sorted_indices[:self.charger_num]
            action[top_k_indices] = 1
        return action.numpy().astype(int)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories):
        states = torch.tensor(np.array([self.state_shaping(t['state']) for t in trajectories]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array([t['reward'] for t in trajectories]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([t['done'] for t in trajectories]), dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(np.array([t['log_prob'].detach().numpy() for t in trajectories]), dtype=torch.float32).to(self.device)

        # Compute values and next values
        values = self.value_net(states).squeeze(-1)
        next_values = torch.cat((values[1:], torch.tensor([0], dtype=torch.float32)))

        # Compute advantages
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)

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

    def save_model(self, model_path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict()
        }, model_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])