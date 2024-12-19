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

		# Read configuration data from the JSON file
		with open(config_fname, "r") as file:
			config = json.load(file)
			
		self.n = config["fleet_size"]  # Number of agents (EVs)
		self.m = config["n_chargers"]  # Number of chargers
		self.k = config["max_time_steps"]  # Maximum number of time steps
		self.delta_t = config["time_step_size"]
		self.h = config["connection_fee($)"]  # Connection fee
		self.max_cap = config["max_cap"]
		self.low_SoC = config["low_SoC"]
		self.initial_SoC = config["initial_SoC"]
		self.d_rate = config["d_rate(%)"]
		self.c_rate = config["c_rate(%)"]
		self.c_r = config["c_r(kWh)"]
		self.rt_bin_edges = config["rt_bin_edges"]
		self.w = config["w"]
		self.rho = config["rho"]
		self.p = config["p"]
		self.data_path = config["data_path"]
		self.ride_time_instance = np.array(config["ride_time_instance"])
		
		self.rt_scenario = config["rt_scenario"]
		self.charging_scenario = config["charging_scenario"]
		self.initial_SoC_scenario = config["initial_SoC_scenario"]
		self.payment_rates_24hrs = config["payment_rates_data($)"][self.rt_scenario]
		self.assign_probs_24hrs = config["order_assign_data"][self.rt_scenario+f"_{self.delta_t}"]
		self.rt_probs = config["ride_time_probs_data"]['probabilities'][self.rt_scenario]
		self.charging_price_24hrs = config["charging_price($/kWh)"]
		self.config = config

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


	def seed(self, seed_value=None):
		"""Set the random seed for reproducibility."""
		self.np_random = np.random.RandomState(seed_value)  # Create a new random state
		random.seed(seed_value)  # Seed Python's built-in random
		return [seed_value]


	def summarize_env(self):
		
		summary = {
				"Environment Info": {
					"Number of EVs in the Fleet": self.n,
					"Total Number of Chargers": self.m,
					"Total Time Steps": self.k,
					"Time Range": f"{self.config['t_0']} to {self.config['t_T']}",
					"Fee for Connecting to Charger (USD)": self.h,
					"Battery Capacity of Each EV (kWh)": self.max_cap,
					"SoC Consumed Per Step (%)": self.d_rate,
					"SoC Charged Per Step (%)": self.c_rate,
					"Low Battery Threshold (SoC)": self.low_SoC,
					f"Probability of Receiving Ride Orders within {self.delta_t} Minutes": self.assign_probs_24hrs,
					"Hours Sorted by Probability of Receiving Ride Orders": np.argsort(self.assign_probs_24hrs)[::-1].tolist(),
				},
				"Ride Info": {
					"Discretized Ride Time Probability Distribution": dict(zip(self.rt_bin_edges, self.rt_probs)),
					"Unit Step Ride Order Payment Rate (USD)": self.payment_rates_24hrs,
					"Hour of Maximum Payment Rate": np.argmax(self.payment_rates_24hrs),
					"Hour of Minimum Payment Rate": np.argmin(self.payment_rates_24hrs),
					"Hours Sorted by Payment per Step (Max to Min)": np.argsort(self.payment_rates_24hrs)[::-1].tolist(),
				},
				"Charging Price Info": {
					"Charging Price (USD/kWh)": self.charging_price_24hrs,
					"Hour of Maximum Charging Price (USD)": np.argmax(self.charging_price_24hrs),
					"Hour of Minimum Charging Price (USD)": np.argmin(self.charging_price_24hrs),
					"Hours Sorted by Charging Price (Max to Min)": np.argsort(self.charging_price_24hrs)[::-1].tolist(),
				}
			}


		# Convert the summary into a readable text format
		summary_str = ""
		for category, data in summary.items():
			summary_str += f"{category}:\n"
			for key, value in data.items():
				summary_str += f"  - {key}: {value}\n"
			summary_str += "\n"
		return summary_str


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
		# state["SoC"] = np.random.uniform(0, 1, size=self.n).round(3)
		state["SoC"] = self.initial_SoC

		self.obs = state  # Store it as the environment's state

		# Empty trajectories
		self.trajectories = {'RideTime': np.zeros((self.n, self.k+1)),
							 'ChargingStatus': np.zeros((self.n, self.k+1)),
							 'SoC': np.zeros((self.n, self.k+1)),
							 'actions': np.zeros((self.n, self.k), dtype=int),
							 'rewards': []} # ride time, charging status, state of charge, action
		
		
		# return the observation
		return self.obs


	def is_zero(self, x):

		return 1 if x == 0 else 0


	def ride_time_generator(self):

		bin_centers = [(self.rt_bin_edges[i] + self.rt_bin_edges[i + 1]) / 2 for i in range(len(rt_bin_edges) - 1)]

		ride_times = []
		for i in range(self.n):
			if np.random.random() < self.rho[self.obs["TimeStep"][i]]:
				bin_index = np.random.choice(len(bin_centers), size=1, p=self.rt_probs)  # Choose a bin index
				bin_index = bin_index[0]  # np.random.choice returns an array
				rt_in_minutes = np.random.uniform(low=self.rt_bin_edges[bin_index], high=self.rt_bin_edges[bin_index + 1])  # Generate random ride time
				
				# Exponentiate and discretize the ride time to steps
				ride_time = int(np.exp(rt_in_minutes) / self.delta_t)  
			else:
				ride_time = 0

			ride_times.append(int(ride_time)) 

		return ride_times
	
	def get_agent_state(self, agent_index):

		return (self.obs["TimeStep"][agent_index],
				self.obs["RideTime"][agent_index],
				self.obs["ChargingStatus"][agent_index],
				self.obs["SoC"][agent_index])


	def feasible_action(self, actions: list[int]) -> bool:
		# If actions in feasible return True, else return str to explain why action is infeasible.
		# if xxxxx:
			# return f"actions[{i}] if feasible because fleet is on ride"
		# if xxx:
			# return f"The number of charging fleets exceed xxx"
		for i in range(self.n):
			state_t = self.get_agent_state(i)
			action_t = actions[i]
			
			if state_t[0] >= 2:
				assert action_t == 0, (
				f"Agent {i}: Action must be 0 if ride leading time >= 2."
			)
			
			if state_t[2] >= 1.:
				assert action_t == 0, (
					f"Agent {i}: Continuing to charge would exceed battery capacity."
				)
	
		assert sum(actions) <=self.m,  (
			f"Total charging exceeds available chargers at time step."
		)
				
		return True

	def step(self, actions):

		# Assert that it is a valid action
		assert self.action_space.contains(actions), "Invalid Action"
		feasible = self.feasible_action(actions)
		if isinstance(feasible, str):
			raise BaseException(feasible)

		current_step = self.obs["TimeStep"][0]
		# random_ride_times = self.ride_time_generator()
		

		sum_rewards = 0
		for i in range(self.n):

			t, rt, ct, SoC = self.get_agent_state(i)
			action = actions[i]
			random_ride_times = self.ride_time_instance[i, t]

			next_SoC = SoC + ct * self.c_rate[i] + (1-ct) * (-self.d_rate[i])

			if action == 0:
				if SoC <= self.low_SoC:
					order_time = 0
				else:
					order_time = np.minimum(random_ride_times, int(SoC/self.d_rate[i]))
		
			
			if rt >= 2 and ct == 0:
				# Active ride scenario
				# (ride_time, charg_time) transitions to (ride_time-1,0)
				# Payment handled at trip start, so reward is 0
				next_state = (rt - 1, 0, next_SoC)
				reward = 0

			elif rt == 1 and ct == 0:
				if action == 0:
					# print("about to finish ride, start taking orders.")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("about to finish ride, start charging.")
					next_state = (0, 1, next_SoC)
					reward = -self.h - self.p[t] * self.c_r[i]
			   

			elif rt == 0 and ct > 0:
				# Charging scenario
				# (ride_time, charg_time) transitions from (0, >0) to (0, a) dependent on a

				if action == 0:
					# print("start taking orders")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("continue charging.")
					next_state = (0, 1, next_SoC)
					reward = - self.p[t] * self.c_r[i]

			elif rt == 0 and ct== 0: # Idle state
				
				if action == 0:
					# print("start taking orders.")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("start charging.")
					next_state = (0, 1, next_SoC)
					reward = -self.h - self.p[t] * self.c_r[i]

			else:
				raise ValueError("This condition should never occur.")
				
			self.obs["TimeStep"][i] = t + 1
			self.obs["RideTime"][i] = next_state[0]
			self.obs["ChargingStatus"][i] = next_state[1]
			self.obs["SoC"][i] = np.maximum(np.minimum(next_state[2], 1.0), 0.)
			sum_rewards += reward

			# print(f'state, action {(rt, ct, SoC), action}')
			# print(f'next state {next_state}')
			# print(f"agent {i} has reward {reward}.")
			# print("\n")

		# Increment the episodic return: no discount factor for now
		self.ep_return += sum_rewards

		# save trajectories
		# next_step = current_step+1
		self.trajectories['actions'][:,current_step] = actions
		self.trajectories['RideTime'][:,current_step+1] = self.obs["RideTime"]
		self.trajectories['ChargingStatus'][:,current_step+1] = self.obs["ChargingStatus"]
		self.trajectories['SoC'][:,current_step+1] =self.obs["SoC"]
		self.trajectories['rewards'].append(sum_rewards)

		# If all values in the first column are equal to k, terminate the episode
		done = np.all(self.obs["TimeStep"] == self.k)

		return self.obs, reward, done, []


	def render(self):

		# i = np.random.randint(0, self.n-1)
		for i in range(self.n):
			print('Show trajectory of agent %d ......' % i)
			# agents = random.sample(range(self.n), 3)

			ride_times = self.trajectories['RideTime'][i,1:]
			fractions_of_cap = self.trajectories['SoC'][i,1:] # to range [0,1]
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
	def __init__(self, config_fname: str):
		self.env = RoadCharging(config_fname)
		super().__init__(self.env)
	# def __init__(self, env):
		# super().__init__(env)

	def action(self, action):
		for i in range(self.n):
			if self.obs["RideTime"][i] >= 1: # if on a ride, not charge
				action[i] = 0
			elif self.obs["SoC"][i] > 1-self.c_rate[i]: # if full capacity, not charge
				action[i] = 0
			elif self.obs["SoC"][i] <= self.low_SoC: # if low capacity has to charge
				action[i] = 1

		total_charging_requests = sum(1 for a, s in zip(action, self.obs["ChargingStatus"]) if s == 0 and a == 1)
		total_continue_charging = sum(1 for a, s in zip(action, self.obs["ChargingStatus"]) if s == 1 and a == 1)
		# released_charger = sum(1 for a, s in zip(action, self.obs["ChargingStatus"]) if s == 1 and a == 0)

		if total_charging_requests + total_continue_charging > self.m: # limit charging requests to available charging capacity
			print('Exceed charger capacity!')
			# charging_requests = sum(action)
			# available_capacity = self.m - sum(self.obs["ChargingStatus"])
			continue_agents = [i for i, (a, s) in enumerate(zip(action, self.obs["ChargingStatus"])) if s == 1 and a == 1]
			requesting_agents = [i for i, (a, s) in enumerate(zip(action, self.obs["ChargingStatus"])) if s == 0 and a == 1]

			available_capacity = self.m - total_continue_charging

			if available_capacity <= 0:
				print('No charger available now.')

				to_flip = requesting_agents
				for i in to_flip:
					action[i] = 0

			elif available_capacity > 0:

				if np.any(action == 1):
					# Scheme #1:
					# Randomly select from the set of agents requesting charging and set their charging actions to 0
					to_flip = random.sample(requesting_agents, total_charging_requests-available_capacity)
					# Scheme #2:
					# sort charging agents based on their SoC from low to high
					# battery_level = dict()
					# for i in charging_agents:
					#     battery_level[i] = self.obs['SoC'][i]

					# sorted_battery_level = dict(sorted(battery_level.items(), key=lambda item: item[1]))
					# print('sorted_battery_level:', sorted_battery_level)
					# to_flip = list(sorted_battery_level.keys())[self.m:]

					# print('Agents requesting charging:', requesting_agents)
					# print('Flip agents:', to_flip)

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


def main():

	data_file = "D://ORLLM//repo//road_charging//output//road_charging//data//test_data//configuration//"

	env = ConstrainAction(RoadCharging(data_file+"config_default_instance_1.json"))

	env.summarize_env()
	env.seed(42)

	# Number of steps you run the agent for
	num_steps = env.k

	# Reset the environment to generate the first observation
	obs = env.reset()

	for step in range(num_steps):
		# # this is where you would insert your policy:
		# take random action
		action = env.action_space.sample()

		# step (transition) through the environment with the action
		# receiving the next observation, reward and if the episode has terminated or truncated
		obs, reward, done, info = env.step(action)
		# print('next ride time', obs["RideTime"])
		# print('next charging status', obs["ChargingStatus"])

		# If the episode has ended then we can reset to start a new episode
		if done:
			print('Final SoC:', env.obs['SoC'])
			# Render the env
			env.render()
			obs = env.reset()


	# save results
	# with open(env.save_path+'saved_trajectories.pkl', 'wb') as f:
	#     pickle.dump(env.trajectories, f)

	# Close the env
	env.close()


if __name__ == "__main__":
	main()
