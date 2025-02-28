import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy
import os

from env_marl import MultiAgentRoadCharging  # 你的多智能体环境

# ------------------ 1) Replay Buffer ------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, actions_arbitrated, rewards, next_state, done):
        """
        注意: 这里的 `rewards` 将会是塑形后的奖励, 而不是原始(真实)奖励
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

# ------------------ 2) Networks (Actor/Critic) ------------------
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出[0,1], 表示"充电"动作的概率
        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=64):
        """
        state_dim: 全局状态维度
        n_agents: 智能体数量
        """
        super(CriticNetwork, self).__init__()
        # Critic输入: state + all_agents_actions => total_dim = state_dim + n_agents
        self.input_dim = state_dim + n_agents
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, actions):
        """
        state: (batch_size, state_dim)
        actions: (batch_size, n_agents) in [0,1]
        """
        x = torch.cat([state, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------ 3) MADDPG Agent ------------------
class MADDPGAgent:
    def __init__(self, 
                 actor_lr, critic_lr, 
                 obs_dim, state_dim, 
                 n_agents, agent_index,
                 gamma=0.95, tau=0.01, hidden_dim=64,
                 device='cpu'):
        self.agent_index = agent_index
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.device = device
        
        # Actor
        self.actor = ActorNetwork(obs_dim, hidden_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic
        self.critic = CriticNetwork(state_dim, n_agents, hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs, exploration=False):
        """
        obs: shape (obs_dim,) ，其中 obs[3] 是 SOC
        返回离散动作 {0,1}
        """
        soc = obs[3]
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            p = self.actor(obs_t).item()  # 得到充电概率
        self.actor.train()
        
        # 自适应探索：若 SOC 低，则增加探索充电的概率
        if exploration:
            # 默认 eps 为 0.05，但如果 SOC 低则设为更高，比如 0.5
            eps = 0.05 if soc >= 0.5 else 0.5
            if np.random.rand() < eps:
                # 若处于低 SOC 情况下，可以直接返回充电（1）
                return 1 if soc < 0.5 else np.random.randint(0, 2)
        # 正常情况下根据 actor 输出的概率采样：伯努力采样
        action = 1 if np.random.rand() < p else 0
        return action

    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_targets(self):
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

# ------------------ 4) 仲裁函数 ------------------
def arbitrate_actions(actions, m, soc_list=None, mode='random'):
    """
    参数:
      actions: list of {0,1}, 长度 n
      m: 可用充电桩数量
      soc_list: 如果非 None，则为各 EV 当前 SOC 的列表（长度 n）
      mode: 'random' 或 'priority'
    """
    a_tilde = actions[:]  # 拷贝
    total_requests = sum(a_tilde)
    if total_requests <= m:
        return a_tilde
    
    idx_ones = [i for i, val in enumerate(a_tilde) if val == 1]
    
    if mode == 'priority' and soc_list is not None:
        # 按 SOC 升序排序，SOC 低的优先
        idx_ones_sorted = sorted(idx_ones, key=lambda i: soc_list[i])
        chosen = idx_ones_sorted[:m]
    elif mode == 'random':
        chosen = np.random.choice(idx_ones, m, replace=False)
    else:
        # 默认顺序
        chosen = idx_ones[:m]
    
    # 将所有充电请求先置 0，再将选中的设置为 1
    for i in idx_ones:
        a_tilde[i] = 0
    for c in chosen:
        a_tilde[c] = 1
    return a_tilde



# ------------------ 辅助函数：对 SoC过低做惩罚（举例） ------------------
def shape_rewards(obs_list, next_obs_list, real_rewards, soc_threshold=0.5, penalty=5.0, bonus=5.0):
    """
    改进后的奖励塑形函数：
    - 如果 EV 当前 SOC < soc_threshold 且下一状态 SOC 得到改善，则给予 bonus 奖励；
    - 如果 SOC 没有改善甚至下降，则额外扣分。
    
    参数:
      obs_list: 当前各 EV 的观测（列表，每个元素 shape=(4,)）
      next_obs_list: 下一时刻各 EV 的观测
      real_rewards: 环境返回的真实奖励（数组或列表）
    """
    shaped_rewards = real_rewards.copy()
    for i in range(len(obs_list)):
        soc_now = obs_list[i][3]
        soc_next = next_obs_list[i][3]
        delta = soc_next - soc_now  # SOC 变化
        # 当 SOC 很低时，希望充电能获得奖励
        if soc_now < soc_threshold:
            if delta > 0:
                # 如果充电导致 SOC 提升，则给予正向奖励（比例可调）
                shaped_rewards[i] += bonus * delta
            else:
                # 如果没有充电，SOC 未提升甚至下降，则额外惩罚
                shaped_rewards[i] -= penalty * abs(delta)
    return shaped_rewards



# ------------------ 5) 训练更新函数 ------------------
def train_maddpg_agent(agent_i, agents, replay_buffer, batch_size):
    agent = agents[agent_i]
    device = agent.device
    
    # 采样
    state_b, actions_b, actions_arbi_b, rewards_b, next_state_b, done_b = replay_buffer.sample(batch_size)
    # 转tensor
    state_t = torch.FloatTensor(state_b).to(device)            # (B, state_dim)
    actions_arbi_t = torch.FloatTensor(actions_arbi_b).to(device)  # (B, n_agents)
    # 注意: 这里 rewards_b 已经是"塑形后"的奖励
    shaped_rewards_t = torch.FloatTensor(rewards_b[:, agent_i]).unsqueeze(-1).to(device)  # (B,1)
    
    next_state_t = torch.FloatTensor(next_state_b).to(device)  # (B, state_dim)
    done_t = torch.FloatTensor(done_b).unsqueeze(-1).to(device)# (B,1)
    
    # ======== Critic Update ========
    with torch.no_grad():
        # 下个时刻: 所有agent用target_actor输出 [0,1] => next_actions(连续)
        next_actions_list = []
        for idx, ag in enumerate(agents):
            # 假设 state_dim = n_agents * obs_dim，obs_dim=4
            obs_dim_ = ag.actor.fc1.in_features  # 4
            start = idx * obs_dim_
            end = (idx+1) * obs_dim_
            obs_i_t = next_state_t[:, start:end]  # shape (B, obs_dim)

            p_next = ag.target_actor(obs_i_t).squeeze(-1)  # (B,)
            next_actions_list.append(p_next)
        
        next_actions_t = torch.stack(next_actions_list, dim=1)  # (B, n_agents)
        
        # 计算 target Q
        Q_next = agent.target_critic(next_state_t, next_actions_t)
        y = shaped_rewards_t + agent.gamma * (1 - done_t) * Q_next
    
    Q_now = agent.critic(state_t, actions_arbi_t)
    critic_loss = F.mse_loss(Q_now, y)
    
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()
    
    # ======== Actor Update ========
    # 让 agent_i 的 actor 输出新的(连续)动作, 其余agent保持不变或detach
    cur_actions_list = []
    for idx, ag in enumerate(agents):
        obs_dim_ = ag.actor.fc1.in_features
        start = idx * obs_dim_
        end = (idx+1) * obs_dim_
        obs_i_t = state_t[:, start:end]
        
        if idx == agent_i:
            p_cur = ag.actor(obs_i_t).squeeze(-1)  # (B,)
        else:
            with torch.no_grad():
                p_cur = ag.actor(obs_i_t).squeeze(-1)
        cur_actions_list.append(p_cur)
    cur_actions_t = torch.stack(cur_actions_list, dim=1)  # (B, n_agents)
    
    # 在Critic中评分
    Q_val = agent.critic(state_t, cur_actions_t)
    actor_loss = -Q_val.mean()
    
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    agent.actor_optimizer.step()


# ------------------ 6) 主训练循环(修改部分) ------------------
def maddpg_train(env, 
                 n_agents, 
                 n_episodes=50,
                 max_steps=200,
                 m=1,
                 gamma=0.95, tau=0.01,
                 actor_lr=1e-3, critic_lr=1e-3,
                 batch_size=64,
                 buffer_capacity=100000,
                 print_interval=10,
                 device='cpu'):
    """
    env: MultiAgentRoadCharging (返回真实reward)
    n_agents: 智能体数量
    ...
    重点: 训练时对 reward 做塑形, 但统计时用原 reward
    """
    obs_dim = 4
    state_dim = env.state_dim
    # 初始化代理
    agents = []
    for i in range(n_agents):
        agent_i = MADDPGAgent(actor_lr, critic_lr, 
                              obs_dim, state_dim, 
                              n_agents, i, 
                              gamma=gamma, tau=tau, 
                              device=device)
        agents.append(agent_i)
    
    # 建立经验回放
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    all_rewards = []  # 存储每回合(真实)奖励
    for ep in range(n_episodes):
        obs_list = env.reset()  # [obs_0,..., obs_{n-1}], each shape(4,)
        ep_reward = np.zeros(n_agents, dtype=float)  # 记录真实收益
        
        for t in range(max_steps):
            # 1) 多智能体选择动作
            actions = []
            for i in range(n_agents):
                a_i = agents[i].select_action(obs_list[i], exploration=True)
                actions.append(a_i)
            
            # 2) 仲裁 => 保证同时充电数 <= m
            # 获取各 EV 的 SOC（假设 obs 中第4个元素为 SOC）
            soc_list = [obs[3] for obs in obs_list]
            # 2) 仲裁：使用优先模式，让 SOC 较低的 EV 优先充电
            actions_arbi = arbitrate_actions(actions, m, soc_list=soc_list, mode='priority')
            
            # 3) 拼装全局state (obs_0||obs_1||...||obs_{n-1}), shape(4*n_agents)
            state_concat = np.concatenate(obs_list, axis=0)
            
            # 4) 与环境交互(返回真实收益)
            next_obs_list, real_rewards, done, info = env.step(actions_arbi)
            
            # --> 做奖励塑形 <--
            shaped_rewards = shape_rewards(obs_list,next_obs_list, real_rewards, soc_threshold=0.5, penalty=5.0)
            
            # 5) 再拼 next_state
            next_state_concat = np.concatenate(next_obs_list, axis=0)
            
            # 将 "shaped_rewards" 存进replay, 用于RL更新
            # 但 ep_reward 加的是 real_rewards (真实收益)
            replay_buffer.push(
                state_concat, 
                actions, 
                actions_arbi, 
                shaped_rewards,  # 写入的是塑形后奖励
                next_state_concat, 
                done
            )
            ep_reward += real_rewards  # 累计真实收益
            
            # 6) 训练更新
            if len(replay_buffer) > batch_size:
                for i in range(n_agents):
                    train_maddpg_agent(i, agents, replay_buffer, batch_size)
                for ag in agents:
                    ag.update_targets()
            
            obs_list = next_obs_list
            if done:
                break
        
        # 一回合结束, 记录并打印
        all_rewards.append(ep_reward)
        
        if (ep+1) % print_interval == 0:
            avg_r = np.mean(all_rewards[-print_interval:], axis=0)
            print(f"Episode {ep+1}/{n_episodes}, avg reward per agent = {avg_r} | avg reward:{np.mean(avg_r)}")
    
    return agents, all_rewards


# # ------------------ 主函数示例 ------------------
# if __name__ == "__main__":
#     config_file = "config1_5EVs_1chargers.json"
#     env = MultiAgentRoadCharging(config_file)
#     n_agents = env.n

#     trained_agents, rewards_history = maddpg_train(
#         env, 
#         n_agents=n_agents, 
#         n_episodes=20,
#         max_steps=50,
#         m=1,
#         device='cpu'
#     )

#     print("训练结束, 全部回合真实收益:")
#     print(rewards_history)
#     env.render()  # 将结果画图或保存


# ============ 用法示例 ============
if __name__ == "__main__":
    # 假设你有一个配置文件 config1_5EVs_1chargers.json
    n_EVs = 5
    n_chargers = 1
    avg_return = 0
    SoC_data_type = "high"
    data_folder = "test_cases"
    results_folder = "results"
    policy_name = "base_policy"
    instance_count = 20

    instance_num = 1
    
    
    test_case = f"all_days_negativePrices_{SoC_data_type}InitSoC_{n_chargers}for{n_EVs}"
    test_cases_dir = os.path.join(data_folder, test_case)  
    data_file = os.path.join(test_cases_dir, f"config{instance_num}_{n_EVs}EVs_{n_chargers}chargers.json")
    env = MultiAgentRoadCharging(data_file)

    n_agents = env.n
    device = 'cpu'  # or 'cuda' if GPU available
    

    # 进行简单训练
    trained_agents, rewards_history = maddpg_train(
        env, 
        n_agents=n_agents, 
        n_episodes=100,   # 迭代回合
        max_steps=96,    # 每回合最大步数
        m=1,             # 同时充电限制
        batch_size=32,
        device=device
    )

    print("训练结束，rewards_history = ", rewards_history)
    env.render()
