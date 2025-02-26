import os
import sys
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from agent import PPOAgent
from dataset import ExpertDataset
from config import Config
from wrapper import RoadChargingWrapper
from torch.utils.data import DataLoader
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from env.gym_env import RoadCharging


def test(agent: PPOAgent, config: Config) -> list[float]:
    results = []
    for data_file in os.listdir(config.test_dir):
        env = RoadCharging(config_fname=os.path.join(config.test_dir, data_file))
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
        results.append(env.ep_return)
    return sum(results) / len(results)


def train_sl(config: Config, dataset: ExpertDataset):
    bc_net = config.policy_net(config.state_length, config.ev_num).to(config.device)
    dataloader = DataLoader(dataset, batch_size=config.bc_batch_size, shuffle=True)
    optimizer = optim.Adam(bc_net.parameters(), lr=config.bc_lr)

    num_ones = sum(sum(action) for _, action in dataset)
    num_zeros = len(dataset) * config.ev_num - num_ones
    pos_weight = torch.tensor([num_zeros / (num_ones + 1e-5)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_result = 0
    losses = []
    agent = PPOAgent(config)
    for episode in range(config.bc_train_episodes):
        for state, action in dataloader:
            state = torch.tensor(state, dtype=torch.float32, device=config.device)
            action = torch.tensor(action, dtype=torch.float32, device=config.device)
            optimizer.zero_grad()
            predicted_action = bc_net(state)
            loss = criterion(predicted_action, action)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        if episode % config.bc_test_frequency == 0:            
            agent.policy_net = bc_net.to(config.device)
            result = test(agent, config)
            print(f"episode: {episode}, average train loss: {sum(losses) / len(losses)}, average test return: {result}")
            agent.save_model(f"bc_{episode}.pkl")
            if result > best_result:
                agent.save_model("bc_best.pkl")
    return os.path.join(config, "bc_best.pkl")


def train_rl(config: Config, pretrained_model_path: str=None):
    env = RoadCharging(config.demo_data_file)
    agent = PPOAgent(config)
    if pretrained_model_path:
        agent.load_model(pretrained_model_path)

    expert_prob = config.init_expert_prob
    experience_buffer = []
    for episode in range(config.rl_train_episodes):
        state = env.reset()
        done = False
        train_returns = []

        while not done:
            action, log_prob = agent.select_action_with_exploring(state)
            next_state, reward, done, _ = env.step(action)
            experience_buffer.append({'state': state, 'action': action, 'reward': reward, 'log_prob': log_prob, 'done': done})
            state = next_state
        train_returns.append(env.ep_return)
        
        if len(experience_buffer) >= config.buffer_size:
            agent.update(experience_buffer)
            experience_buffer = []

        if episode % config.rl_test_frequency == 0:
            expert_prob *= config.expert_prob_decay
            result = test(agent, config)
            agent.save_model(f"ppo_{episode}.pkl")
            print(f"episode: {episode}, train return: {sum(train_returns) / len(train_returns)}, average test return: {result}")

if __name__ == "__main__":
    config = Config()

    dataset = ExpertDataset(config)
    dataset.collect_expert_data()

    pretrained_model_path = train_sl(config, dataset)
    train_rl(config, pretrained_model_path)
