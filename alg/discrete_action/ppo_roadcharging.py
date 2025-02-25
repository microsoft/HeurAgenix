import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
import random


import gym
from gym.spaces import Discrete, Box, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

DEBUG = False
# TRAIN = False
TRAIN = True
EV_NUM = 5

if TRAIN:
    from env.gym_env import RoadCharging
    # from env.modified_env import ConstrainAction
else:
    # from env.origin_env import ConstrainAction
    # from env.gym_env import ConstrainAction
    from env.gym_env import RoadCharging


# Parameters
# gamma = 1
gamma = 0.99
render = False
seed = 1
# seed = 0
log_interval = 10

if EV_NUM == 5:
    # data_file = "env/data/config1_5EVs_1chargers.json"
    data_file = "env/data/test_cases/all_days_negativePrices_highInitSoC_1for5/config1_5EVs_1chargers.json"
elif EV_NUM == 10:
    data_file = "env/data/test_cases/all_days_negativePrices_highInitSoC_1for5/config1_10EVs_1chargers.json"
elif EV_NUM == 8:
    data_file = "env/data/test_cases/all_days_negativePrices_highInitSoC_1for8/config1_8EVs_1chargers.json"

# env = ConstrainAction(data_file)
env = RoadCharging(data_file)
env.summarize_env()
env.seed(seed)

# env = gym.make('CartPole-v1').unwrapped
num_state = env.observation_space_dim
# num_state = sum([space.shape[0] if isinstance(space, Box) else space.n for space in env.observation_space.spaces.values()])
# num_action = pow(2, env.action_space.n)
num_action = env.action_space.n + 1
num_charger = env.m


print("num_state:",num_state)
print("num_action:",num_action)

torch.manual_seed(seed)
env.action_space.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

def decimal_to_binary_array(action, num_bits):
    # 使用 bin() 转换为二进制字符串，去掉 '0b' 前缀，填充到 num_bits 位
    binary_string = bin(action)[2:].zfill(num_bits)
    # 将二进制字符串转换为数组
    binary_array = [int(bit) for bit in binary_string]
    return binary_array
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, num_action)
        # self.fc_k_logits = nn.Linear(256, num_charger)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.action_head.weight)

    def forward(self, x):
        # # 打印权重
        # print("fc1 weights:", self.fc1.weight)
        # print("fc2 weights:", self.fc2.weight)
        # # print("action_head weights:", self.action_head.weight)

        # 检查权重是否包含 nan 或 inf
        if torch.any(torch.isnan(self.fc1.weight)) or torch.any(torch.isinf(self.fc1.weight)):
            print("fc1 weights contain invalid values")
        if torch.any(torch.isnan(self.fc2.weight)) or torch.any(torch.isinf(self.fc2.weight)):
            print("fc2 weights contain invalid values")
        if torch.any(torch.isnan(self.action_head.weight)) or torch.any(torch.isinf(self.action_head.weight)):
            print("action_head weights contain invalid values")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_prob = F.softmax(self.action_head(x), dim=1)
        # action_prob = torch.sigmoid(self.action_head(x))  # 得到 [0,1] 之间的概率
        # print("x", x)
        # print("action_prob:", action_prob)
        
        return action_prob




class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.state_value = nn.Linear(128, 1)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.state_value.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 1e-4)

        if not os.path.exists('param'):
            os.makedirs('param/net_param')
            os.makedirs('param/img')

    def process_state(self, state):
        # print("state:",state)
        # print("num_state:",num_state)

        if isinstance(state, np.ndarray):
            # If state is already a NumPy array, return it directly
            return state
        timestep = np.array(state['TimeStep'])
        ridetime = np.array(state['RideTime'])
        ridetime_max = ride_time_max = np.max(env.ride_time_instance)
        ridetime_normalized = ridetime / ridetime_max

        timestep_max = env.k
        timestep_normalized = timestep / timestep_max

        # print("ridetime_normalized:",ridetime_normalized)
        # print("ridetimemax:",ridetime_max)

        
        charging_status = np.array(state['ChargingStatus'])
        soc = np.array(state['SoC'])
        # state = np.concatenate([timestep, ridetime, charging_status, soc])
        state = np.concatenate([ridetime_normalized, charging_status, soc])
        # state = np.concatenate([timestep_normalized, ridetime_normalized, charging_status, soc])

        

        return state
    
    # def process_state(self, state):
    #     if isinstance(state, np.ndarray):
    #         # If state is already a NumPy array, return it directly
    #         return state
    #     timestep = np.array(state['TimeStep'])
    #     ridetime = np.array(state['RideTime'])
    #     ridetime_max = ride_time_max = np.max(env.ride_time_instance)
    #     ridetime_normalized = ridetime / ridetime_max

    #     current_state = np.array(state['CurrentState'])

    #     timestep_max = env.k
    #     timestep_normalized = timestep / timestep_max

    #     # print("ridetime_normalized:",ridetime_normalized)
    #     # print("ridetimemax:",ridetime_max)

        
    #     charging_status = np.array(state['ChargingStatus'])
    #     soc = np.array(state['SoC'])
    #     # state = np.concatenate([timestep, ridetime, charging_status, soc])
    #     # state = np.concatenate([ridetime_normalized, charging_status, soc])
    #     state = np.concatenate([timestep_normalized, ridetime_normalized, charging_status, soc])

    #     return state

    def select_action(self, state, deterministic=False, activation="softmax"):

        # 生成mask
        mask = torch.ones(1, env.n)
        for i in range(env.n):
            if state["RideTime"][i] >= 1: # if on a ride, not charge
                mask[0][i] = 0
            elif state["SoC"][i] > 1-env.c_rates[i]: # if full capacity, not charge
                mask[0][i] = 0
        
        if activation == "discrete_action":
            mask = torch.cat((mask, torch.tensor([[1]])), dim=1)


        state = self.process_state(state)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0)

        if activation == "softmax":
            with torch.no_grad():
                action_prob, k = self.actor_net(state)
                k = k.item()

        elif activation == "sigmoid":
            with torch.no_grad():
                action_prob = self.actor_net(state)

        elif activation == "discrete_action":
            with torch.no_grad():
                action_prob = self.actor_net(state)

        # 对 action_prob 进行掩码操作
        # print("mask:",mask)
        # print("action_prob:",action_prob)
        action_prob = action_prob * mask
        # 重新归一化
        action_prob_sum = action_prob.sum(dim=1, keepdim=True)
        action_prob = action_prob / action_prob_sum
        action_prob = action_prob.squeeze()

        # print("action_prob:",action_prob)


        if torch.all(action_prob == 0):
            # print("All probabilities are zero. Returning default action.")
            action = [0 for _ in range(action_prob.size(0))]
            return action, 1
        
        if activation == "sigmoid":
            if deterministic:
                # 选择具有最高概率的动作
                action = torch.round(action_prob).view(-1).tolist() 

                # 计算 action 中值为 1 的个数
                num_ones = sum(action)
                if num_ones > env.m:
                    # 如果 1 的个数超过 m，则使用 topk 选择前 m 个概率最大的动作
                    topk_indices = torch.topk(action_prob, env.m).indices
                    action = [0] * len(action_prob)
                    for idx in topk_indices:
                        action[idx] = 1
                prob = action_prob
                # return action, action_prob
            else:
                # 采样动作
                m = Bernoulli(action_prob)
                action = m.sample()
                # 计算 action 中值为 1 的个数
                num_ones = action.sum().item()
                if num_ones > env.m:
                    # 如果 1 的个数超过 m，则按概率采样 m 个动作
                    sampled_indices = torch.multinomial(action_prob, env.m, replacement=False)
                    action = torch.zeros_like(action_prob)
                    action[sampled_indices] = 1

                # print("action prob:",action_prob)
                prob = 1
                for i in range(len(action)):
                    prob *= action_prob[i] if action[i] == 1 else (1 - action_prob[i])
            return action, prob
            # return action, m.log_prob(action).sum().item()
        
        
        if activation == "discrete_action":
            if deterministic:
                # 选择具有最高概率的动作
                # action_index = torch.argmax(action_prob, dim=1)
                action_index = torch.argmax(action_prob)
            else:
                # 采样动作
                c = Categorical(action_prob)
                action_index = c.sample()

            action_prob_selected = action_prob[action_index]
            action = [0 for _ in range(action_prob.size(0) - 1)]
            if action_index < env.n:
                action[action_index] = 1
            else:
                action = [0 for _ in range(env.n)]

            # print("action:",action)
            # print("action_prob:",action_prob)
            # print(action_prob_selected)
            return action, action_prob_selected
        
        elif activation == "softmax":
            if deterministic:
                # 选择具有最高概率的动作
                # action = torch.round(action_prob).view(-1).tolist() 
                action = self.select_topk_action(action_prob, k)
            else:
                # 采样动作
                sampled_action = torch.multinomial(action_prob, k, replacement=False)  # replacement=False 表示不允许重复

                action = [0 for _ in range(action_prob.size(1))]
                for i in sampled_action:
                    action[i] = 1
            
            prob = 1
            for i in range(len(action)):
                if action[i] == 1:
                    prob *= action_prob[0][i]
                # prob *= action_prob[0][i] if action[i] == 1 else 1 - action_prob[0][i]
            # print("action:",action)
            # print("action_prob:",prob)
            return action, prob
        else:
            raise ValueError("Invalid activation function")
        return
    
    def select_topk_action(self, action_prob, k=1, threshold=0.5):
        # 选择概率最大的 k 个动作
        topk_action = torch.topk(action_prob, k, dim=1)
        topk_action_indices = topk_action.indices
        topk_action_prob = topk_action.values
        # # 过滤概率小于阈值的动作
        # topk_action_indices = topk_action_indices[topk_action_prob > threshold]
        # topk_action_prob = topk_action_prob[topk_action_prob > threshold]
        
        action = [0 for _ in range(action_prob.size(1))]
        for i in topk_action_indices:
            action[i] = 1
        return action
    
    def action_constraint(self, action):
        array_sum = sum(action)

        # array_sum = action.sum().item()  # 使用 PyTorch 的 sum 方法并转换为标量值
        # print("array_sum:",array_sum)
        # print("action:",action)
        # if array_sum > env.m:
        #     action = [0 for _ in range(len(action))]
        # return action

        if array_sum > env.m:
            # 找出数组中所有的1的索引
            ones_indices = [i for i, bit in enumerate(action) if bit == 1]
            # 随机选择k个1的索引
            selected_indices = random.sample(ones_indices, env.m)
            for i in range(len(action)):
                if i in selected_indices:
                    action[i] = 1
                else:
                    action[i] = 0
            return action
        else:
            return action


    def get_value(self, state):
        state = self.process_state(state)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'param/net_param/actor_net' + str(time.time())[:10] + '.pkl')
        torch.save(self.critic_net.state_dict(), 'param/net_param/critic_net' + str(time.time())[:10] + '.pkl')
    
    def load_param(self, filepath='param/net_param/actor_net' + str(time.time())[:10] + '.pkl'):
        self.actor_net.load_state_dict(torch.load(filepath))
        print(f'Model parameters loaded from {filepath}')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        # for t in self.buffer:
        #     print("t:",t.state)

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!

                # action_prob, k = self.actor_net(state[index])
                action_prob = self.actor_net(state[index])
                action_prob = action_prob.gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

def smooth(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def train():
    agent = PPO()

    reward_list = []

    for i_epoch in range(5000):
        state = env.reset()
        # state = agent.process_state(state)
        # print("state:",state)

        if render: env.render()

        total_reward = 0
        total_action_ones = 0

        for t in count():
            action, action_prob = agent.select_action(state, activation="discrete_action")
            state_normalized = agent.process_state(state)

            if DEBUG == True:
                print("action:",action)
                print("action_prob:",action_prob)

            action = agent.action_constraint(action)
            action = np.array(action)

            num_ones = np.sum(action)
            total_action_ones += num_ones


            result = env.step(action)
            # print("step" ,result)
            # print("action_prob:",action_prob)

            next_state, reward, done, _ = result
            next_state_normalized = agent.process_state(next_state)
            total_reward += reward

            trans = Transition(state_normalized, action, action_prob, reward, next_state_normalized)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state
            
            # print("reward:",reward)

            if done :
                print("epoch:",i_epoch)
                print("total_action_ones:",total_action_ones)
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
        
        print("total_reward:",total_reward)
        print("env.stoch_step", env.stoch_step)
        reward_list.append(total_reward)
        total_reward = 0

    window_size = 10

    # 平滑数据
    smoothed_rewards = smooth(reward_list, window_size)

    # 绘制图表
    plt.figure()
    plt.plot(smoothed_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Epoch (Smoothed)')
    plt.tight_layout()
    plt.savefig('param/img/total_reward2.png')  # 保存图片
    plt.close()  # 关闭图表以释放内存


    agent.save_param()
    return

def test_network():
    agent = PPO()
    agent.load_param('alg/discrete_action/param/net_param/actor_net1-10-5000-1739863846.pkl')  # 加载模型参数
    # 使用模型进行推理
    # state = torch.tensor([...])  # 输入状态
    state = env.reset()
    state_normalized = agent.process_state(state)

    soc_values = []
    actions = []
    total_reward = 0

    for t in count():
        
        action, action_prob = agent.select_action(state, deterministic = True, activation="discrete_action")

        action = agent.action_constraint(action)
        action = np.array(action)
        # print("action:",action)
        # print("action_prob:",action_prob)
        actions.append(action)

        result = env.step(action)
        # print("step" ,result)

        next_state, reward, done, _ = result

        soc_list = [f"{x:.3f}" for x in next_state['SoC']]
        soc_list = [round(float(x), 3) for x in next_state['SoC']]  # 将字符串转换为浮点数并限制小数位数
        
        for i in range(len(soc_list)):
            SoC = state['SoC'][i]
            next_SoC = next_state['SoC'][i]
            rt = state['RideTime'][i]
            ct = state['ChargingStatus'][i]
            c_rate = env.c_rates[i]
            d_rate = env.d_rates[i]
            # print("i:", i, "action:", action[i], "t:", "SoC:", f"{SoC:.2f}", "next_SoC:", f"{next_SoC:.2f}", "rt:", rt, "ct:", ct)
            print("i:", i, "t:", "SoC:", f"{SoC:.2f}", "next_SoC:", f"{next_SoC:.2f}", "rt:", rt, "ct:", ct)

        print(soc_list)
        print("action_prob", action_prob)
        print("action:",action)

        soc_values.append(soc_list)
        
        # next_state = agent.process_state(next_state)

        if render: env.render()
        state = copy.deepcopy(next_state)
        print("reward:",reward)
        total_reward += reward

        if done :
            # print("epoch:",t)
            # if len(agent.buffer) >= agent.batch_size:agent.update(t)
            # agent.writer.add_scalar('liveTime/livestep', t, global_step=t)
            break

    # print("soc_values:",soc_values)
    for value in soc_values:
        print(value)

    print("total_reward:",total_reward)

    # soc_values = soc_values[:100]
    # for i in range(len(soc_values)):
    #     print(soc_values[i][0])

    # Plotting the SoC values
    # 确保目录存在
    output_dir = f'results/config1-{EV_NUM}ev-1charger'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting the SoC values and actions

    for i in range(EV_NUM):
        fig, ax1 = plt.subplots()

        # 绘制 SoC
        ax1.plot([soc[i] for soc in soc_values], label=f'SoC {i}', color='b')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('SoC', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # 创建第二个 y 轴
        ax2 = ax1.twinx()
        ax2.plot([action[i] for action in actions], label=f'Action {i}', color='r', linestyle='--')
        ax2.set_ylabel('Action', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # 添加图例
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        plt.title('State of Charge (SoC) and Actions over Time')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'soc_action_{i}.png'))  # 保存图片
        plt.close()  # 关闭图表以释放内存

    return

if __name__ == '__main__':
    if TRAIN:
        train()
    else:
        test_network()
    # train()
    # print("end")

    # test_network()

    # action = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
    # agent = PPO()
    # agent.action_constraint(action, 2)