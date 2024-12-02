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
import os

from gym import Env, spaces


class RoadCharging(Env):
    def __init__(self, config_fname: str):
        super(RoadCharging, self).__init__()

        # Load configuration data from the JSON file
        with open(config_fname, "r") as file:
            config = json.load(file)

        # Store the configuration
        self.config = config
        self.delta_t = config["step_length"]

        # Extract relevant configuration parameters
        self.fleet_size = config['fleet_size']  # Number of EVs in the fleet
        self.total_chargers = config['total_chargers']  # Total number of chargers
        self.max_time_steps = int(config["time_horizon"] / self.delta_t)
        self.connection_fee = config['connection_fee']  # $ per connection session
        self.assign_prob = pd.read_csv(config['prob_fpath']).iloc[:, 0].tolist()  # Probability of receiving a ride order when idle
        self.max_cap = config['max_cap']  # Max battery capacity (kWh)
        self.consume_rate = round(1 / config["time_SoCfrom100to0"] * self.delta_t, 3)  # Battery consumption rate per time step
        self.charger_speed = round(1 / config["time_SoCfrom0to100"] * self.delta_t, 3)  # Charger speed per time step

        # Load data files for various parameters
        self.RT_mean = pd.read_csv(config["trip_time_fpath"][0]).iloc[:, 0].tolist()
        self.RT_std = pd.read_csv(config["trip_time_fpath"][1]).iloc[:, 0].tolist()
        self.order_price = pd.read_csv(config["trip_fare_fpath"]).iloc[:, 0].tolist()
        self.charging_price = pd.read_csv(config["charging_price_fpath"]).iloc[:, 0].tolist()

        plt.plot(self.charging_price)
        plt.show()
        # Assign values to class attributes
        self.n = self.fleet_size  # Number of agents (EVs)
        self.m = self.total_chargers  # Number of chargers
        self.k = self.max_time_steps  # Maximum number of time steps
        self.h = self.connection_fee  # Connection fee
        self.rho = np.repeat(self.assign_prob, int(60 / self.delta_t)) # Ride order assignment probability
        self.max_cap = self.max_cap  # Max capacity of EVs
        self.consume_rate = [self.consume_rate] * self.n  # Battery consumption rate per time step
        self.charger_speed = [self.charger_speed] * self.n  # Charger speed per time step
        self.mu = np.repeat(self.RT_mean, int(60 / self.delta_t))  # Ride time mean
        self.sigma = np.repeat(self.RT_std, int(60 / self.delta_t))  # Ride time standard deviation
        self.w = np.repeat(self.order_price, int(60 / self.delta_t)) * self.delta_t  # Order price per time step
        self.r = self.charger_speed * self.max_cap  # Charger rate (kWh per time step)
        self.p = np.repeat(self.charging_price, int(60 / self.delta_t))  # Charging price per time step
        self.rng = np.random.default_rng()  # Random number generator
        self.low_battery = 0.1  # Low battery threshold
        self.ride_time_distribution_name = config["ride_time_distribution_name"]

        # Save path for results
        self.save_path = config['save_path']

        # Check if the path exists, and create it if it doesn't
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"Directory '{self.save_path}' created.")
        else:
            print(f"Directory '{self.save_path}' already exists.")

        # Observation space: n agents, each with 4 state variables
        self.observation_shape = (self.n, 4)

        # Define the observation space for each agent
        self.observation_space = spaces.Dict({
            "TimeStep": spaces.MultiDiscrete([self.k + 1] * self.n),  # Time step for each agent (0 to k)
            "RideTime": spaces.MultiDiscrete([self.k + 1] * self.n),  # Ride time for each agent (0 to k)
            "ChargingStatus": spaces.MultiDiscrete([2] * self.n),  # Charging status: 0 (not charging) or 1 (charging)
            "SoC": spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=float),  # State of Charge (0 to 1)
        })

        # Action space: n agents, each can take a binary action (0 or 1)
        self.action_space = spaces.MultiBinary(self.n)

        # Debug prints to verify some of the key values
        # print(f'Consume Rate: {self.consume_rate}')
        # print(f'Charger Speed: {self.charger_speed}')
        # print(f'Order Prices Length: {len(self.w)}')



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
    

    def summarize_env(self):
        summary = {
            "Environment Info": {
            "Number of EVs in the Fleet": self.fleet_size,
            "Total Number of Chargers": self.total_chargers,
            "Total Time Horizon (Steps)": self.max_time_steps,
            "Connection Fee (USD)": self.connection_fee,
            "Battery Capacity of Each EV (kWh)": self.max_cap,
            "Energy Consumed Per Step (kWh)": self.max_cap * self.consume_rate[0],
            "Energy Charged Per Step (kWh)": self.max_cap * self.charger_speed[0],
            "Low Battery Threshold (SoC)": self.low_battery,
            "Probability of Receiving Ride Orders at Each Hour": self.assign_prob,
            "Hours Sorted by Probability of Receiving Ride Orders": np.argsort(self.assign_prob)[::-1] 
        },
        }

        if self.ride_time_distribution_name == "log-normal":
            summary["Ride Info"] = {
                "Ride Time Distribution Type": "Log-normal",
                "Mean of Logged Ride Times at Each Hour (in time steps)": self.RT_mean,
                "Std Dev of Logged Ride Times at Each Hour (in time steps)": self.RT_std,
                "Hour of Maximum Ride Time Mean": np.argmax(self.RT_mean),  # Index of max value
                "Hour of Minimum Ride Time Mean": np.argmin(self.RT_mean),  # Index of min value
                "Hours Sorted by Ride Time Mean (Max to Min)": np.argsort(self.RT_mean)[::-1],
                "Hour of Maximum Ride Time Std Dev": np.argmax(self.RT_std),  # Index of max value
                "Hour of Minimum Ride Time Std Dev": np.argmin(self.RT_std),  # Index of min value
                "Hours Sorted by Ride Time Std Dev (Max to Min)": np.argsort(self.RT_std)[::-1],
                "Ride Order Payment per Step at Each Hour (USD)":self.order_price,
                "Hour of Maximum Payment": np.argmax(self.order_price),  # Index of max value
                "Hour of Minimum Payment": np.argmin(self.order_price),  # Index of min value
                "Hours Sorted by Payment per Step (Max to Min)": np.argsort(self.order_price)[::-1],
            }

        summary["Charging Price Info"] = {
            "Charging Price at Each Hour (USD)": self.charging_price,
            "Hour of Maximum Charging Price (USD)": np.argmax(self.charging_price),  # Index of max price
            "Hour of Minimum Charging Price (USD)": np.argmin(self.charging_price),  # Index of min price
            "Hours Sorted by Charging Price (Max to Min)": np.argsort(self.charging_price)[::-1],
        }

        # Convert the summary into a readable text format
        summary_str = ""
        for category, data in summary.items():
            summary_str += f"{category}:\n"
            for key, value in data.items():
                summary_str += f"  - {key}: {value}\n"
            summary_str += "\n"

        # Print the summary to the console
        print(summary_str)

        # Save the summary to a text file
        with open("environment_summary.txt", "w") as file:
            file.write(summary_str)


    def get_action_meanings(self):
        return {0: "Available for taking ride orders", 1: "Go to charge"}
    
    def get_operational_status(self):
        # Determine operational status for all agents
        operational_status = []  # List to store the status of each vehicle

        for i in range(self.n): 
            if self.obs["RideTime"][i] == 0 and self.obs["ChargingStatus"][i] == 0:
                status = "Idle"
            elif self.obs["RideTime"][i] > 0 and self.obs["ChargingStatus"][i] == 0:
                status = "Ride"
            elif self.obs["RideTime"][i] == 0 and self.obs["ChargingStatus"][i] == 1:
                status = "Charge"
            else:
                raise ValueError(f"Unexpected state for agent {i}: "
                         f"RideTime={self.obs['RideTime'][i]}, "
                         f"ChargingStatus={self.obs['ChargingStatus'][i]}")  # Raise an error
    
            operational_status.append((i, status))  # Append (agent_id, status) to the list

        return operational_status


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
                if self.ride_time_distribution_name == "log-normal":
                    ride_time = int(self.rng.lognormal(self.mu[t], self.sigma[t])/self.delta_t) # convert to time steps

            ride_time = np.minimum(ride_time, int(bLevel/self.consume_rate[agent]))

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
                if np.random.random() < self.rho[state["TimeStep"]]:
                    alpha = 0
                else:
                # next_state is a random number drawn from get_ride_time() with prob 1-0.01
                    alpha = ride_time

        # get next charging status
        beta = action

        # get next state of charge
        theta = (1-action) * (state["SoC"]-self.consume_rate[agent_idx]) + action * (state["SoC"]+self.charger_speed[agent_idx])
        theta = round(np.minimum(np.maximum(theta, 0), 1.0), 3)

        self.obs["TimeStep"][agent_idx] = state["TimeStep"] + 1
        self.obs["RideTime"][agent_idx] = alpha
        self.obs["ChargingStatus"][agent_idx] = beta
        self.obs["SoC"][agent_idx] = theta



    def step(self, action):

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        current_step = self.obs["TimeStep"][0]
        self.trajectories['actions'][:,current_step] = action

        # Reward for executing a step.
        alpha = self.obs["RideTime"]
        beta = self.obs["ChargingStatus"]
        tau = self.ride_time_generator()

        reward = sum(self.w[self.obs["TimeStep"][i]] * tau[i] * (1-action[i]) * self.is_zero(alpha[i]) * (1-beta[i])
                     - self.h * action[i] * (1-beta[i])
                     - self.p[self.obs["TimeStep"][i]] * self.r[i] * action[i]
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
