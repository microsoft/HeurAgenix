import copy
import torch

class MADDPGAgent:
    def __init__(self, 
                 actor_lr, critic_lr, 
                 obs_dim, state_dim, 
                 n_agents, agent_index,
                 gamma=0.95, tau=0.01, hidden_dim=64):
        """
        agent_index: 第 i 个智能体的索引，用于一些区分
        """
        self.agent_index = agent_index
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        
        # Actor
        self.actor = ActorNetwork(obs_dim, hidden_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic
        self.critic = CriticNetwork(state_dim, n_agents, hidden_dim)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs, exploration=False):
        """
        obs: ndarray shape (obs_dim,) => 转为tensor后forward
        返回: 0/1 的离散动作(可先输出概率p再采样)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)  # shape (1, obs_dim)
        with torch.no_grad():
            p = self.actor(obs_t).item()  # scalar in [0,1]
        if exploration:
            # 简易做法: 加点随机扰动
            eps = 0.05  # 可调
            if np.random.rand() < eps:
                return np.random.randint(0,2)  # 0 or 1
        # 按概率p进行伯努力采样
        action = 1 if np.random.rand() < p else 0
        return action
    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_targets(self):
        """Soft update both actor & critic"""
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
