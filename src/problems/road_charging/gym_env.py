import numpy as np
import gym
import random
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import pandas as pd
import json

from gym import Env, spaces


class RoadCharging(Env):
    def __init__(self, config_fname:str):
        super(RoadCharging, self).__init__()

        with open(config_fname, "r") as file:
            config = json.load(file)

        self.config = config
        self.delta_t = config["step_length"]

        fleet_size = config['fleet_size'] # size of EV fleet
        total_chargers = config['total_chargers'] # number of total chargers
        max_time_steps = int(config["time_horizon"]/self.delta_t)
        connection_fee = config['connection_fee'] # ($ per connection session)
        assign_prob = config['assign_prob'] # Probability of receiving a ride order when the vehicle is in idle status
        max_cap = config['max_cap'] # (kWh)
        consume_rate = round(1/config["time_SoCfrom100to0"]*self.delta_t, 3) # (percentage per time step, a fully charged battery can sustain 8 hours)
        charger_speed = round(1/config["time_SoCfrom0to100"]*self.delta_t, 3) # (percentage per time step)
        # load data
        RT_mean = pd.read_csv(config["trip_time_fpath"][0]).iloc[:,0].tolist()
        RT_std = pd.read_csv(config["trip_time_fpath"][1]).iloc[:,0].tolist()
        order_price = pd.read_csv(config["trip_fare_fpath"]).iloc[:,0].tolist()
        charging_price = pd.read_csv(config["charging_price_fpath"]).iloc[:,0].tolist()

        self.n = fleet_size
        self.m = total_chargers
        self.k = max_time_steps
        self.h = connection_fee
        self.assign_prob = assign_prob
        self.max_cap = max_cap
        self.mu = np.repeat(RT_mean, int(60/self.delta_t))
        self.sigma = np.repeat(RT_std, int(60/self.delta_t))
        self.save_path = config['save_path']
        self.consume_rate = consume_rate
        self.charger_speed = charger_speed
        self.w = np.repeat(order_price, int(60/self.delta_t)) * self.delta_t
        self.r = charger_speed * max_cap # kWh per time step
        self.p = np.repeat(charging_price, int(60/self.delta_t))
        self.rng = np.random.default_rng()
        self.low_battery = 0.1

        # Observation space: n agents, each has 4 state variables: time step, ride time, charging status, SoC
        self.observation_shape = (self.n, 4)

        self.observation_space = spaces.Dict({
            "TimeStep": spaces.MultiDiscrete([self.k+1] * self.n),  # n dimensions, each in range 0 to k
            "RideTime": spaces.MultiDiscrete([self.k+1] * self.n),  # n dimensions, each in range 0 to k
            "ChargingStatus": spaces.MultiDiscrete([2] * self.n),      # n dimensions, each in range 0 to 1
            "SoC": spaces.Box(low=0.0, high=1.0, shape=(self.n, ), dtype=float),  # Continuous Box with n elements
        })


        # Action space: n agents, each takes action 0 or 1
        self.action_space = spaces.MultiBinary(self.n)

        print('consume_rate:', self.consume_rate)
        print('charger_speed:', self.charger_speed)
        print('len(w):', len(self.w))


    def seed(self, seed_value=None):
        """Set the random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed_value)  # Create a new random state
        random.seed(seed_value)  # Seed Python's built-in random
        return [seed_value]


    def get_sample_price(self, quarter):

        df = pd.read_csv(self.config['additional']["all_chargingPrice_fpath"])

        if quarter == 'Q1':
            unique_dates = pd.read_csv(self.config['additional']["Q1_dates_fpath"])['Local Date'].tolist()
        elif quarter == 'Q2':
            unique_dates = pd.read_csv(self.config['additional']["Q2_dates_fpath"])['Local Date'].tolist()
        elif quarter == 'Q3':
            unique_dates = pd.read_csv(self.config['additional']["Q3_dates_fpath"])['Local Date'].tolist()
        elif quarter == 'Q4':
            unique_dates = pd.read_csv(self.config['additional']["Q4_dates_fpath"])['Local Date'].tolist()

        random_date = np.random.choice(unique_dates)

        p_t = df[df['Local Date']==random_date]['SP-15 LMP'].to_numpy()

        return np.repeat(p_t, int(60/self.delta_t))


    def get_action_meanings(self):
        return {0: "Available for taking ride orders", 1: "Go to charge"}


    def reset(self):

        # Reset the reward
        self.ep_return  = 0

        # Reset the observation
        state = {
            "TimeStep": np.zeros(self.n, dtype=int),
            "RideTime": np.zeros(self.n, dtype=int),
            "ChargingStatus": np.zeros(self.n, dtype=int),
            "SoC": np.zeros(self.n, dtype=float),
        }

        # Initialize battery SoC randomly
        state["SoC"] = np.random.uniform(0, 1, size=self.n).round(3)

        self.obs = state  # Store it as the environment's state

        # Empty trajectories
        self.trajectories = {'states': np.zeros((self.n, 3, self.k+1)),
                             'actions': np.zeros((self.n, self.k), dtype=int),
                             'rewards': []} # ride time, charging status, state of charge, action

        self.p = self.get_sample_price('Q2')

        # return the observation
        return self.obs


    def is_zero(self, x):

        return 1 if x == 0 else 0


    def ride_time_generator(self):

        ride_times = []

        for agent, bLevel in enumerate(self.obs["SoC"]):

            t = self.obs["TimeStep"][agent]

            if bLevel <= self.low_battery:
                ride_time = 0
            else:
                ride_time = int(self.rng.lognormal(self.mu[t], self.sigma[t])/self.delta_t) # convert to time steps

            ride_time = np.minimum(ride_time, int(bLevel/self.consume_rate))

            ride_times.append(ride_time)

        return ride_times


    def agent_step(self, agent_idx, ride_time, state, action):

        # get next ride time
        if action == 1:
            alpha = 0
        elif action == 0:
            if state["RideTime"] > 0:
                alpha = np.maximum(state["RideTime"]-1, 0)
            elif state["ChargingStatus"] > 0:
                alpha = 0
            elif state["RideTime"] == 0 and state["ChargingStatus"] == 0:
                # next_state is zero with prob 0.01
                if np.random.random() < self.assign_prob:
                    alpha = 0
                else:
                # next_state is a random number drawn from get_ride_time() with prob 1-0.01
                    alpha = ride_time

        # get next charging status
        beta = action

        # get next state of charge
        theta = (1-action) * (state["SoC"]-self.consume_rate) + action * (state["SoC"]+self.charger_speed)
        theta = round(np.minimum(np.maximum(theta, 0), 1.0), 3)

        self.obs["TimeStep"][agent_idx] = state["TimeStep"] + 1
        self.obs["RideTime"][agent_idx] = alpha
        self.obs["ChargingStatus"][agent_idx] = beta
        self.obs["SoC"][agent_idx] = theta



    def step(self, action):

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        print('Current SoC:', self.obs['SoC'])
        print('Current action:', action)

        current_step = self.obs["TimeStep"][0]
        self.trajectories['actions'][:,current_step] = action

        # Reward for executing a step.
        alpha = self.obs["RideTime"]
        beta = self.obs["ChargingStatus"]
        tau = self.ride_time_generator()

        reward = sum(self.w[self.obs["TimeStep"][i]] * tau[i] * (1-action[i]) * self.is_zero(alpha[i]) * (1-beta[i])
                     - self.h * action[i] * (1-beta[i])
                     - self.p[self.obs["TimeStep"][i]] * self.r * action[i]
                     for i in range(self.n))

        # apply the action: modify self.obs
        for i in range(self.n):

            agent_state = {"TimeStep": self.obs["TimeStep"][i],
                           "RideTime": self.obs["RideTime"][i],
                           "ChargingStatus": self.obs["ChargingStatus"][i],
                           "SoC": self.obs["SoC"][i]}

            agent_action = action[i]

            self.agent_step(i, tau[i], agent_state, agent_action)

        # Increment the episodic return: no discount factor for now
        self.ep_return += reward

        # save trajectories

        next_step = current_step+1
        self.trajectories['states'][:,0,next_step] = self.obs["RideTime"]
        self.trajectories['states'][:,1,next_step] = self.obs["ChargingStatus"]
        self.trajectories['states'][:,2,next_step] =self.obs["SoC"]
        self.trajectories['rewards'].append(reward)

        # If all values in the first column are equal to k, terminate the episode
        done = np.all(self.obs["TimeStep"] == self.k)

        return self.obs, reward, done, []


    def render(self):

        i = np.random.randint(0, self.n-1)
        print('Show trajectory of agent %d ......' % i)
        # agents = random.sample(range(self.n), 3)

        ride_times = self.trajectories['states'][i,0,1:]
        fractions_of_cap = self.trajectories['states'][i,2,1:] # to range [0,1]
        actions = self.trajectories['actions'][i,:]


        _, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))
        # First plot
        ax1.step(range(self.k), ride_times, color='blue', linestyle='-', label='ride times')
        ax1.set_ylabel('Remaining Ride Time Steps', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks
        ax1.legend(loc="upper right")
        ax1.set_xlabel('Time step')

        # Second plot
        ax2.step(range(self.k), fractions_of_cap, color='black', linestyle='-.', label='state of charge')
        ax2.set_ylabel('State of Charge', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc="upper left")

        # Create a secondary y-axis for the second plot
        ax2_secondary = ax2.twinx()
        ax2_secondary.step(range(self.k), actions, color='red', linestyle='-', label='actions')
        ax2_secondary.set_ylabel('Actions', color='red')
        ax2_secondary.tick_params(axis='y', labelcolor='red')
        ax2_secondary.yaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks
        ax2_secondary.legend(loc="upper right")
        ax2.set_xlabel('Time step')


        plt.tight_layout()
        plt.show()

class ConstrainAction(gym.ActionWrapper):
    def __init__(self, config: str):
        env = RoadCharging(config)
        super()._init_(env)
    
    # def __init__(self, env):
    #     super().__init__(env)

    def action(self, action):
        for i in range(self.n):
            if self.obs["RideTime"][i] >= 1: # if on a ride, not charge
                action[i] = 0
            elif self.obs["SoC"][i] > 1-self.charger_speed: # if full capacity, not charge
                action[i] = 0
            elif self.obs["SoC"][i] <= self.low_battery: # if low capacity has to charge
                action[i] = 1

        if sum(action) + sum(self.obs["ChargingStatus"]) >= self.m: # limit charging requests to available charging capacity
            print('Exceed charger capacity!')
            charging_requests = sum(action)
            available_capacity = self.m - sum(self.obs["ChargingStatus"])
            charging_agents = np.where(action == 1)[0] if np.any(action == 1) else []

            if available_capacity <= 0:
                print('No charger available now.')
                # flip all, including those with low capacity. which will not be assigned any
                # order. they will wait until a charger becomes available
                # If their battery level drops to 0, it will remain at zero, and they will continue to make requests to charge at 
                # each decision epoch until a charger becomes available
                # This policy may lead to low overall returns, which GPT should learn to avoid.
                # Alternatively, we can use a sorting strategy. this can guarantee no EV will drop to zero battery. but
                # this may not be very fair for vehicles that planned to charge earlier well before
                # their battery levels reached a low point.

                to_flip = charging_agents
                action[to_flip] = 0

            elif available_capacity > 0:

                if np.any(action == 1):
                    # Scheme #1:
                    # Randomly select from the set of agents requesting charging and set their charging actions to 0
                    to_flip = random.sample(list(charging_agents), charging_requests-available_capacity)
                    # Scheme #2:
                    # sort charging agents based on their SoC from low to high
                    # battery_level = dict()
                    # for i in charging_agents:
                    #     battery_level[i] = self.obs['SoC'][i]

                    # sorted_battery_level = dict(sorted(battery_level.items(), key=lambda item: item[1]))
                    # print('sorted_battery_level:', sorted_battery_level)
                    # to_flip = list(sorted_battery_level.keys())[self.m:]

                    print('Agents requesting charging:', charging_agents)
                    print('Agents in charging:', self.obs["ChargingStatus"])
                    print('Flip agents:', to_flip)

                    action[to_flip] = 0

        # for i in range(self.n): # if SoC is too low, must charge | Q: What if you swap it with a vehicle that has a low SoC as well?
        #     if self.obs["SoC"][i] <= 0.1 and action[i]==0:
        #         print('checkpoint 3')
        #         # Swap the action of agent i with a randomly selected agent that takes action 1
        #         # charging_agents = np.where(action == 1)[0] if np.any(action == 1) else []
        #         # if np.any(action == 1):
        #         #     j = np.random.choice(charging_agents)

        #         #     action[j] = 0
        #         #     action[i] = 1

        #         # assuming a backup charger is available, but at an extremely high charging price
        #         action[i] = 1

        return action

