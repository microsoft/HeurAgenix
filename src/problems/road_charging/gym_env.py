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

        self.fleet_size = config['fleet_size']  # Number of EVs in the fleet
        self.total_chargers = config['total_chargers']  # Total number of chargers
        self.max_time_steps = int(config["time_horizon"] / self.delta_t)
        self.connection_fee = config['connection_fee']  # $ per connection session
        self.assign_prob = pd.read_csv(config['prob_fpath']).iloc[:, 0].tolist()  # Probability of receiving a ride order when idle
        self.max_cap = config['max_cap']  # Max battery capacity (kWh)
        self.consume_rate = round(1 / config["time_SoCfrom100to0"] * self.delta_t, 3)  # Battery consumption rate per time step
        self.charger_speed = round(1 / config["time_SoCfrom0to100"] * self.delta_t, 3)  # Charger speed per time step

        self.ride_time_bins = config["ride_time_bins"]
        self.ride_time_probs = config["ride_time_probs"]
        self.order_price = pd.read_csv(config["trip_fare_fpath"]).iloc[:, 0].tolist()
        self.charging_price = pd.read_csv(config["charging_price_fpath"]).iloc[:, 0].tolist()

        self.n = self.fleet_size  # Number of agents (EVs)
        self.m = self.total_chargers  # Number of chargers
        self.k = self.max_time_steps  # Maximum number of time steps
        self.h = self.connection_fee  # Connection fee
        self.rho = np.repeat(self.assign_prob, int(60 / self.delta_t)) # Ride order assignment probability
        self.max_cap = self.max_cap  # Max capacity of EVs
        self.consume_rate = [self.consume_rate] * self.n  # Battery consumption rate per time step
        self.charger_speed = [self.charger_speed] * self.n  # Charger speed per time step
        self.w = np.repeat(self.order_price, int(60 / self.delta_t)) * self.delta_t  # Order price per time step
        self.r =  [x * self.max_cap for x in self.charger_speed] # Charger rate (kWh per time step)
        self.p = np.repeat(self.charging_price, int(60 / self.delta_t))  # Charging price per time step
        self.rng = np.random.default_rng()  # Random number generator
        self.low_battery = 0.1  # Low battery threshold

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

        # self.get_sample_price("Q2")



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

        # p_t = df[df['Local Date']==random_date]['SP-15 LMP'].to_numpy().tolist()
        p_t = df[df['Local Date']==random_date]['SP-15 LMP'].to_numpy()
        print('debug get_sample_price:', len(p_t))
        print('view p_t:', p_t)
        plt.plot(p_t)
        plt.show()

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

        summary["Ride Info"] = {
            "Discretized Ride Time Probability Distribution": dict(zip(self.ride_time_bins, self.ride_time_probs)),
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

        # self.p = self.get_sample_price('Q2')

        # return the observation
        return self.obs


    def is_zero(self, x):

        return 1 if x == 0 else 0


    def ride_time_generator(self):

        ride_times = []

        # for SoC in self.obs["SoC"]:
        for i in range(self.n):

            if np.random.random() < self.rho[self.obs["TimeStep"][i]]:
                ride_time = int(np.random.choice([rt for rt in self.ride_time_bins if rt > 0],
                                                        p=[prob for prob in self.ride_time_probs if prob>0] ))
            else:
                ride_time = 0

            ride_times.append(int(ride_time)) 

        return ride_times
    
    def get_agent_state(self, agent_index):

        return (self.obs["TimeStep"][agent_index],
                self.obs["RideTime"][agent_index],
                self.obs["ChargingStatus"][agent_index],
                self.obs["SoC"][agent_index])


    def step(self, actions):

        # Assert that it is a valid action
        assert self.action_space.contains(actions), "Invalid Action"

        current_step = self.obs["TimeStep"][0]
        random_ride_times = self.ride_time_generator()

        sum_rewards = 0
        for i in range(self.n):

            t, ride_time, charging_status, SoC = self.get_agent_state(i)
            action = actions[i]
           
            next_SoC = SoC + charging_status * self.charger_speed[i] + (1-charging_status) * (-self.consume_rate[i])

            if ride_time >= 2 and charging_status == 0:
                # Active ride scenario
                # (ride_time, charg_time) transitions to (ride_time-1,0)
                # Payment handled at trip start, so reward is 0
                next_state = (ride_time - 1, 0, SoC-self.consume_rate[i])
                reward = 0

            elif ride_time == 1 and charging_status == 0:
                # Active ride scenario
                # (ride_time, charg_time) transitions to (ride_time-1,0)
                # Payment handled at trip start, so reward is 0
                next_state = (0, action, SoC-self.consume_rate[i])
                reward = action * (-self.h - self.p[t] * self.r[i])

            elif ride_time == 0 and charging_status > 0:
                # Charging scenario
                # (ride_time, charg_time) transitions from (0, >0) to (0, a) dependent on a
                next_state = (0, action, next_SoC)
                reward = action * (- self.p[t] * self.r[i])

            elif ride_time == 0 and charging_status== 0: # Idle state
                
                if action == 0:
                    if SoC <= self.low_battery:
                        ride_time = 0
                    else:
                        ride_time = np.minimum(random_ride_times[i], int(SoC/self.consume_rate[i]))

                    next_state = (ride_time, 0, next_SoC)
                    reward = self.w[t] * ride_time

                elif action == 1:
                    # Start charging from the next step; hence, SoC still drops in the current step due to consumption.
                    # With this logic, if the vehicle is charging at time t, the SoC will increase in the next step,
                    # regardless of whether it decides to continue charging or not at t+1.
                    next_state = (0, 1, next_SoC)
                    reward = action * (-self.h - self.p[t] * self.r[i])

    
            self.obs["TimeStep"][i] = t + 1
            self.obs["RideTime"][i] = next_state[0]
            self.obs["ChargingStatus"][i] = next_state[1]
            self.obs["SoC"][i] = np.maximum(np.minimum(next_state[2], 1.0), 0.)
            sum_rewards += reward

            print(f'next state {next_state}')

            print(f"agent {i} has reward {reward}.")


        # Increment the episodic return: no discount factor for now
        self.ep_return += sum_rewards

        # save trajectories
        # next_step = current_step+1
        self.trajectories['actions'][:,current_step] = actions
        self.trajectories['states'][:,0,current_step+1] = self.obs["RideTime"]
        self.trajectories['states'][:,1,current_step+1] = self.obs["ChargingStatus"]
        self.trajectories['states'][:,2,current_step+1] =self.obs["SoC"]
        self.trajectories['rewards'].append(sum_rewards)

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

