import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, actions_arbitrated, rewards, next_state, done):
        """
        存储一次交互:
        state: np.array 或自定义表示(全局或局部拼接)
        actions: 原始动作 [a^1, a^2, ..., a^n]
        actions_arbitrated: 仲裁后动作 [tilde_a^1, ..., tilde_a^n]
        rewards: list长度n，每个agent的回报 (或统一全局回报也可)
        next_state: 下一个全局状态
        done: 是否episode结束
        """
        self.buffer.append((state, actions, actions_arbitrated, rewards, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, actions, actions_arbitrated, rewards, next_state, done = zip(*batch)
        return (np.array(state), 
                np.array(actions), 
                np.array(actions_arbitrated),
                np.array(rewards), 
                np.array(next_state), 
                np.array(done))
    
    def __len__(self):
        return len(self.buffer)
