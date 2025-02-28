import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层只输出一个值 => 动作概率/分数 => 再映射到 [0,1]
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 映射到 [0,1]
        return x  # 表示"请求充电"的概率 p

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=64):
        """
        state_dim: 全局状态s的维度
        n_agents: 智能体数量
        """
        super(CriticNetwork, self).__init__()
        # Critic输入: state + all_agents_actions (这里动作用浮点[0,1]拼接)
        self.input_dim = state_dim + n_agents
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出Q(s,a^1,...,a^n) 对某个agent的
        
    def forward(self, state, actions):
        """
        state: (batch_size, state_dim)
        actions: (batch_size, n_agents)
        """
        x = torch.cat([state, actions], dim=-1)  # 拼接后输入Critic
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
