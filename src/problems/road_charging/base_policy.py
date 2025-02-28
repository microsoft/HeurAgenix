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
from scipy.stats import lognorm
from gym_env import RoadCharging, ConstrainAction
# from show_trajectory import show_trajectory


def base_policy(env, state):
	alpha = state["RideTime"]
	beta = state["ChargingStatus"]
	theta = state["SoC"]
  
	action = np.zeros(env.n, dtype=int)
	for i in range(env.n):
		if state["RideTime"][i] >= 2: # if on a ride, must not charge
			action[i] = 0
		elif state["SoC"][i] >=1.0: # if full capacity, not charge
			action[i] = 0
		elif state["SoC"][i] <= env.low_SoC: # if low capacity has to charge
			action[i] = 1

	total_start = sum(1 for a, s in zip(action, state["ChargingStatus"]) if s == 0 and a == 1)
	total_continue = sum(1 for a, s in zip(action, state["ChargingStatus"]) if s == 1 and a == 1)
	total_charging = sum(action)
	
	if total_charging > env.m: # limit charging requests to available charging capacity
		print('Exceed charger capacity!')
		requesting_agents = [i for i, (a, s) in enumerate(zip(action, state["ChargingStatus"])) if s == 0 and a == 1]

		available_capacity = env.m - total_continue

		if available_capacity <= 0:
			print('No charger available now.')
			# flip all
			to_flip = requesting_agents
			action[to_flip] = 0

		elif available_capacity > 0:

			if np.any(action == 1):
				to_flip = random.sample(requesting_agents, total_start-available_capacity)
    
				print('Agents requesting charging:', requesting_agents)
				print('Flip agents:', to_flip)

				action[to_flip] = 0

	return action



def main():
	
	n_EVs = 5
	n_chargers = 1
	avg_return = 0
	SoC_data_type = "high"
	data_folder = "test_cases"
	results_folder = "results"
	policy_name = "base_policy"

	instance_count = 20
	for instance_num in range(1, 1+instance_count):
		test_case = f"all_days_negativePrices_{SoC_data_type}InitSoC_{n_chargers}for{n_EVs}"
		test_cases_dir = os.path.join(data_folder, test_case)  
		data_file = os.path.join(test_cases_dir, f"config{instance_num}_{n_EVs}EVs_{n_chargers}chargers.json")
		print(data_file)
		env = ConstrainAction(data_file)
		# env = ConstrainAction(RoadCharging(data_file))
		env.seed(42)

		# Number of agents, states, and actions
		n_steps = env.k
		n_agents = env.n
		n_states = 3  # 3 possible states per agent
		n_actions = 2  # 2 action options per agent

		print(f"Number of agents {env.n}")
		print(f"Number of time steps {env.k}")

		# Training loop
		n_episodes = 1
		ep_return = []
		for episode in range(n_episodes):
			state = env.reset()
			done = False

			while not done:
			
				action = base_policy(env, state)

				# Perform joint actions in the environment
				next_state, rewards, done, _ = env.step(action)

				print(f"return up to now is {env.ep_return}")
				ep_return.append(env.ep_return)

				state = next_state

		solution = {
			"actions": env.trajectory['actions'].tolist(),
			"RideTime": env.trajectory['RideTime'].tolist(),
			"ChargingStatus": env.trajectory['ChargingStatus'].tolist(),
			"SoC": env.trajectory['SoC'].tolist(),
			"final_return": env.ep_return
		}
		save_dir = os.path.join(results_folder, test_case, policy_name)
		os.makedirs(save_dir, exist_ok=True)	
		with open(os.path.join(save_dir, f"instance{instance_num}_solutoin.json"), "w") as f:
			json.dump(solution, f, indent=4)  # Use indent for readability
			
		# show_trajectory(env.n, env.k, env.trajectory, save_dir)
  
		avg_return+=env.ep_return


		# Close the env
		env.close()
	avg_return /= instance_count
	print(f"average return over {instance_count} instances:", avg_return)
	# average return over 20 instances: 3790.3959000000004


if __name__ == "__main__":
	main()
