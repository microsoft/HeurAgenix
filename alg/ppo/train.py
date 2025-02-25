import os
import random
import torch
from agent import PPOAgent
from wrapper import RoadChargingWrapper
from config import Config
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..")
sys.path.insert(0, env_dir)
from env.gym_env import ConstrainAction


def test(agent: PPOAgent, config: Config) -> list[float]:
    results = []
    for data_file in os.listdir(config.test_dir):
        env = ConstrainAction(config_fname=os.path.join(config.test_dir, data_file))
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
        results.append(env.ep_return)
    return sum(results) / len(results)


def train(config: Config):
    env = RoadChargingWrapper(config_fname=config.data_file, mode="train", penalty=config.penalty)
    agent = PPOAgent(env, config)

    expert_prob = config.init_expert_prob
    experience_buffer = []
    for episode in range(config.max_train_episodes):
        state = env.reset()
        done = False

        while not done:
            if random.random() < expert_prob:
                action = config.expert_policy(env, state)
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob = agent.select_action_with_explorer(state)
            next_state, reward, done, _ = env.step(action)
            experience_buffer.append({'state': state, 'action': action, 'reward': reward, 'log_prob': log_prob, 'done': done})
            state = next_state
        
        if len(experience_buffer) >= config.max_buffer_size:
            agent.update(experience_buffer)
            experience_buffer = []

        if episode % config.test_frequency == 0:
            expert_prob *= config.expert_prob_decay
            results = test(agent, config)
            agent.save_model(os.path.join(config.output_dir, f"ppo_{episode}.pkl"))
            print(episode, results)

config = Config()
train(config)