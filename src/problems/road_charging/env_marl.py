# env_marl.py
import numpy as np
import gym
import random
import json
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gym import Env, spaces


class MultiAgentRoadCharging(Env):
    def __init__(self, config_fname: str):
        super(MultiAgentRoadCharging, self).__init__()

        # 1) 读取配置文件
        with open(config_fname, "r") as file:
            config = json.load(file)

        # 2) 提取环境配置
        self.n = config["fleet_size"]    # Number of EVs (agents)
        self.m = config["n_chargers"]    # Number of available chargers
        self.k = config["max_time_step"] # Max time steps in an episode
        self.delta_t = config["time_step_size"]
        self.h = config["connection_fee($)"]
        self.max_cap = config["max_cap"]
        self.low_SoC = config["low_SoC"]
        self.initial_SoCs = config["initial_SoCs"]  # shape: (n,)
        self.d_rates = config["d_rates(%)"]         # length n
        self.c_rates = config["c_rates(%)"]         # length n
        self.c_r = config["charging_powers(kWh)"]   # length n
        self.w = config["w"]      # Payment rate, shape: (k+1,) or so
        self.rho = config["rho"]  # Probability of receiving order
        self.p = config["p"]      # Charging prices
        self.ride_data_instance = np.array(config["ride_data_instance"])  # shape (n, k+1)
        self.config = config

        # 3) 多智能体相关维度
        # 每个智能体的局部观察大小=4,  (TimeStep_i, RideTime_i, ChargingStatus_i, SoC_i)
        self.obs_dim = 4
        # 全局状态可简单定义为 n个agent的局部观测拼接 => 4 * n
        self.state_dim = 4 * self.n

        # 4) 定义action/observation space
        # 单个智能体可选择 action ∈ {0,1}, 这里用 Discrete(2).
        self.action_space = spaces.Discrete(2)
        # 单个智能体的 observation 是长度4, 这里用 Box(...) 做近似表征
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0], dtype=np.float32),
            high=np.array([self.k, self.k, 1, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 5) 运行时记录
        self.ep_return = 0.0
        self.current_step = 0
        self.obs_dict = None
        self.trajectory = None

    def seed(self, seed_value=None):
        np.random.seed(seed_value)
        random.seed(seed_value)
        return [seed_value]

    def reset(self):
        """
        重置环境, 返回多智能体的观测列表: [obs_0, obs_1, ..., obs_{n-1}].
        """
        self.ep_return = 0.0
        self.current_step = 0

        # 初始化状态
        obs_dict = {
            "TimeStep": np.zeros(self.n, dtype=int),
            "RideTime": np.zeros(self.n, dtype=int),
            "ChargingStatus": np.zeros(self.n, dtype=int),
            "SoC": np.zeros(self.n, dtype=float),
        }
        obs_dict["SoC"] = np.array(self.initial_SoCs, dtype=float)
        
        self.obs_dict = obs_dict

        # 轨迹记录
        self.trajectory = {
            'RideTime': np.zeros((self.n, self.k+1), dtype=int),
            'ChargingStatus': np.zeros((self.n, self.k+1), dtype=int),
            'SoC': np.zeros((self.n, self.k+1), dtype=float),
            'actions': np.zeros((self.n, self.k), dtype=int),
            'rewards': []
        }
        self.trajectory['RideTime'][:, 0] = obs_dict["RideTime"]
        self.trajectory['ChargingStatus'][:, 0] = obs_dict["ChargingStatus"]
        self.trajectory['SoC'][:, 0] = obs_dict["SoC"]

        return self._get_obs_list()

    def _get_obs_list(self):
        """
        将 obs_dict 里每个agent的 [TimeStep, RideTime, ChargingStatus, SoC]
        拼装成长度n的列表，每项是 shape(4,) 的np数组.
        """
        obs_list = []
        for i in range(self.n):
            obs_i = np.array([
                self.obs_dict["TimeStep"][i],
                self.obs_dict["RideTime"][i],
                self.obs_dict["ChargingStatus"][i],
                self.obs_dict["SoC"][i]
            ], dtype=np.float32)
            obs_list.append(obs_i)
        return obs_list

    def step(self, actions):
        """
        多智能体step: 
        - actions: list/array, shape=(n,), each in {0,1}
        - return (next_obs_list, reward_list, done, info)
        """
        assert len(actions) == self.n, f"Expected {self.n} actions, got {len(actions)}"
        
        reward_list = np.zeros(self.n, dtype=float)

        for i in range(self.n):
            t_i  = self.obs_dict["TimeStep"][i]
            rt_i = self.obs_dict["RideTime"][i]
            ct_i = self.obs_dict["ChargingStatus"][i]
            SoC_i= self.obs_dict["SoC"][i]

            act_i = actions[i]

            # 简化的可行性检查
            if rt_i >= 2:
                # 还在ride期间 => 强制act=0
                act_i = 0
            if SoC_i >= 1.0 and act_i == 1:
                # 满电不可充
                act_i = 0

            # 计算下一状态
            random_ride_times = self.ride_data_instance[i, t_i]
            next_SoC = np.clip(SoC_i + ct_i*self.c_rates[i] 
                               + (1-ct_i)*(-self.d_rates[i]), 0, 1)

            if rt_i >= 2 and ct_i == 0:
                # ride进行中
                next_rt = rt_i - 1
                next_ct = 0
                r_i = 0.0
            elif rt_i == 1 and ct_i == 0:
                # ride刚结束
                if act_i == 0:
                    # 尝试下一单
                    order_time = self._calc_order_time(SoC_i, random_ride_times, i)
                    next_rt = order_time
                    next_ct = 0
                    r_i = self.w[t_i] * order_time
                else:
                    # 开始充电
                    next_rt = 0
                    next_ct = 1
                    r_i = -self.h - self.p[t_i]*min(self.c_r[i], (1-next_SoC)*self.max_cap)
            elif rt_i == 0 and ct_i > 0:
                # 当前在充电
                if act_i == 0:
                    # 停止充电 => 接单
                    order_time = self._calc_order_time(SoC_i, random_ride_times, i)
                    next_rt = order_time
                    next_ct = 0
                    r_i = self.w[t_i]*order_time
                else:
                    # 继续充电
                    next_rt = 0
                    next_ct = 1
                    r_i = - self.p[t_i]*min(self.c_r[i], (1-next_SoC)*self.max_cap)
            elif rt_i == 0 and ct_i == 0:
                # idle
                if act_i == 0:
                    order_time = self._calc_order_time(SoC_i, random_ride_times, i)
                    next_rt = order_time
                    next_ct = 0
                    r_i = self.w[t_i] * order_time
                else:
                    next_rt = 0
                    next_ct = 1
                    r_i = - self.h - self.p[t_i]*min(self.c_r[i], (1-next_SoC)*self.max_cap)
            else:
                # 其他情况
                raise ValueError("Unexpected state transition.")

            # 更新 env 状态
            self.obs_dict["TimeStep"][i] = t_i + 1
            self.obs_dict["RideTime"][i] = next_rt
            self.obs_dict["ChargingStatus"][i] = next_ct
            self.obs_dict["SoC"][i] = next_SoC

            reward_list[i] = r_i

            # 轨迹记录
            self.trajectory['actions'][i, t_i] = act_i

        # 累积奖励
        sum_rewards = np.sum(reward_list)
        self.ep_return += sum_rewards

        # 更新轨迹
        next_t = self.obs_dict["TimeStep"][0]
        self.trajectory['RideTime'][:, next_t] = self.obs_dict["RideTime"]
        self.trajectory['ChargingStatus'][:, next_t] = self.obs_dict["ChargingStatus"]
        self.trajectory['SoC'][:, next_t] = self.obs_dict["SoC"]
        self.trajectory['rewards'].append(sum_rewards)

        # 是否结束
        done = bool(np.all(self.obs_dict["TimeStep"] >= self.k))
        next_obs_list = self._get_obs_list()

        return next_obs_list, reward_list, done, {}

    def _calc_order_time(self, SoC_i, random_ride_times, i):
        """
        模拟计算新的订单时长
        """
        if SoC_i <= self.low_SoC:
            return 0
        else:
            max_ride = int(round(SoC_i / self.d_rates[i]))
            return min(random_ride_times, max_ride)

    def render(self):
        """
        将各智能体在本回合中的轨迹数据绘制并保存为图片文件，不在屏幕上显示。
        """
    

        # 创建存放图片的文件夹
        save_dir = "render_out"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.n):
            print(f'>> Save trajectory figure of agent {i} ......')

            ride_times = self.trajectory['RideTime'][i, 1:]
            fractions_of_cap = self.trajectory['SoC'][i, 1:]
            actions = self.trajectory['actions'][i, :]

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))
            ax1.step(range(self.k), ride_times, color='blue', label='ride times')
            ax1.set_ylabel('Remaining RideTime')
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.legend(loc='upper right')

            ax2.step(range(self.k), fractions_of_cap, color='black', label='SoC')
            ax2.set_ylabel('State of Charge')
            ax2.legend(loc='upper left')

            ax2_2 = ax2.twinx()
            ax2_2.step(range(self.k), actions, color='red', label='Actions')
            ax2_2.set_ylabel('Actions')
            ax2_2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2_2.legend(loc='upper right')

            ax2.set_xlabel('Time step')
            plt.tight_layout()

            # 将图片保存到文件夹内
            fig.savefig(os.path.join(save_dir, f"agent_{i}_trajectory.png"))
            plt.close(fig)

    def close(self):
        pass
