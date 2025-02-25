import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, time
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import count
import copy
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from env.gym_env import ConstrainAction
from env.gym_env import RoadCharging
# from env.modified_env import ConstrainAction


from torch.distributions import Normal, Categorical, Bernoulli
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self, env, num_state, num_action):
        super(PPO, self).__init__()
        self.actor_net = Actor(num_state, num_action)
        self.critic_net = Critic(num_state)
        self.env = env
        # self.buffer = []
        # self.counter = 0
        # self.training_step = 0
        # self.writer = SummaryWriter('exp')

        # self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-4)
        # self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 1e-4)

        # if not os.path.exists('param'):
        #     os.makedirs('param/net_param')
        #     os.makedirs('param/img')

    def process_state(self, state):
        if isinstance(state, np.ndarray):
            # If state is already a NumPy array, return it directly
            return state
        timestep = np.array(state['TimeStep'])
        ridetime = np.array(state['RideTime'])
        ridetime_max = ride_time_max = np.max(self.env.ride_time_instance)
        ridetime_normalized = ridetime / ridetime_max

        # print("ridetime_normalized:",ridetime_normalized)
        # print("ridetimemax:",ridetime_max)

        
        charging_status = np.array(state['ChargingStatus'])
        soc = np.array(state['SoC'])
        # state = np.concatenate([timestep, ridetime, charging_status, soc])
        state = np.concatenate([ridetime_normalized, charging_status, soc])

        return state
    
    # def process_state(self, state):
    #     # print("state:",state)
    #     # print("num_state:",num_state)

    #     if isinstance(state, np.ndarray):
    #         # If state is already a NumPy array, return it directly
    #         return state
    #     timestep = np.array(state['TimeStep'])
    #     ridetime = np.array(state['RideTime'])
    #     ridetime_max = ride_time_max = np.max(self.env.ride_time_instance)
    #     ridetime_normalized = ridetime / ridetime_max

    #     timestep_max = self.env.k
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
        mask = torch.ones(1, self.env.n)
        for i in range(self.env.n):
            if state["RideTime"][i] >= 1: # if on a ride, not charge
                mask[0][i] = 0
            elif state["SoC"][i] > 1-self.env.c_rates[i]: # if full capacity, not charge
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
                if num_ones > self.env.m:
                    # 如果 1 的个数超过 m，则使用 topk 选择前 m 个概率最大的动作
                    topk_indices = torch.topk(action_prob, self.env.m).indices
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
                if num_ones > self.env.m:
                    # 如果 1 的个数超过 m，则按概率采样 m 个动作
                    sampled_indices = torch.multinomial(action_prob, self.env.m, replacement=False)
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
            if action_index < self.env.n:
                action[action_index] = 1
            else:
                action = [0 for _ in range(self.env.n)]

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

        if array_sum > self.env.m:
            # 找出数组中所有的1的索引
            ones_indices = [i for i, bit in enumerate(action) if bit == 1]
            # 随机选择k个1的索引
            selected_indices = random.sample(ones_indices, self.env.m)
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
        # self.actor_net.load_state_dict(torch.load(filepath))
        self.actor_net.load_state_dict(torch.load(filepath, weights_only=True))
        print(f'Model parameters loaded from {filepath}')




class Actor(nn.Module):
    def __init__(self,num_state,num_action):
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
    def __init__(self, num_state):
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
    
def rl_test(data_path, network_path):
    n_EVs = 10
    n_chargers = 1
    avg_return = 0
    SoC_data_type = "high"
    # data_folder = "test_cases_adjusted"
    data_folder = "env/data/test_cases"
    results_folder = "results_updated"
    policy_name = "base_policy"
    instance_count = 20

    seed = 1
    env = RoadCharging(data_path)
    # env = ConstrainAction(data_path)
    env.seed(seed)

    agent = PPO(env, env.observation_space_dim, env.action_space.n + 1)
    agent.load_param(network_path)  # 加载模型参数
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
        actions.append(action)

        result = env.step(action)

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
            # print("i:", i, "t:", "SoC:", f"{SoC:.2f}", "next_SoC:", f"{next_SoC:.2f}", "rt:", rt, "ct:", ct)

        # print(soc_list)
        # print("action_prob", action_prob)
        # print("action:",action)

        soc_values.append(soc_list)
        
        # next_state = agent.process_state(next_state)

        state = copy.deepcopy(next_state)
        # print("reward:",reward)
        total_reward += reward

        if done :
            # print("epoch:",t)
            # if len(agent.buffer) >= agent.batch_size:agent.update(t)
            # agent.writer.add_scalar('liveTime/livestep', t, global_step=t)
            break

    # print("soc_values:",soc_values)
    # for value in soc_values:
    #     print(value)

    print("total_reward:",total_reward)
    print("ep_return",env.ep_return)
    env.close()
    return env.ep_return



def main():
	
    n_EVs = 5
    n_chargers = 1
    avg_return = 0
    SoC_data_type = "high"
    # SoC_data_type = "polarized"
    # data_folder = "test_cases_adjusted"
    data_folder = "env/data/test_cases"
    results_folder = "results_updated"
    # policy_name = "base_policy"
    if n_EVs == 5:
        # network_path = "alg/discrete_action/param/net_param/actor_net1-5-5000-1739861249.pkl"
        # network_path = "alg/discrete_action/param/net_param/actor_net1740025070.pkl"
        # network_path = "alg/discrete_action/param/net_param/actor_net1-5-5000-sto1740029305.pkl"
        network_path = "alg/discrete_action/param/net_param/actor_net1740035212.pkl"
        

        

    elif n_EVs == 8:
        network_path = "alg/discrete_action/param/net_param/actor_net1-8-5000-1739960890.pkl"
    elif n_EVs == 10:
        network_path = "alg/discrete_action/param/net_param/actor_net1-10-5000-1739863846.pkl"



    instance_count = 20
    all_epreturn = []
    for instance_num in range(1, 1+instance_count):
        test_case = f"all_days_negativePrices_{SoC_data_type}InitSoC_{n_chargers}for{n_EVs}"

        test_cases_dir = os.path.join(data_folder, test_case)  
        data_file = os.path.join(test_cases_dir, f"config{instance_num}_{n_EVs}EVs_{n_chargers}chargers.json")
        epreturn = rl_test(data_file, network_path)
        all_epreturn.append(float(epreturn))
        avg_return += epreturn

    avg_return /= instance_count
    print("EVs,", n_EVs)
    print("Soc data type:", SoC_data_type)
    print("all_epreturn:",all_epreturn)
    print(f"average return over {instance_count} instances:", avg_return)


if __name__ == '__main__':
    main()