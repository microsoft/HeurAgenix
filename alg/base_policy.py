
import json
import random
import os
import numpy as np
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from env.gym_env import RoadCharging, ConstrainAction


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
		# print('Exceed charger capacity!')
		requesting_agents = [i for i, (a, s) in enumerate(zip(action, state["ChargingStatus"])) if s == 0 and a == 1]

		available_capacity = env.m - total_continue

		if available_capacity <= 0:
			# print('No charger available now.')
			# flip all
			to_flip = requesting_agents
			action[to_flip] = 0

		elif available_capacity > 0:
			if np.any(action == 1):
				to_flip = random.sample(requesting_agents, total_start-available_capacity)
				action[to_flip] = 0
		
	return action

def main():
	
	avg_return = 0
	results_folder = "results"
	policy_name = "base_policy"

	instance_count = 20
	test_cases_dir = r"env\data\test_cases\all_days_negativePrices_highInitSoC_1for5"
	for test_case in os.listdir(test_cases_dir):
		data_file = os.path.join(test_cases_dir, test_case)
		env = RoadCharging(data_file)
		env.seed(42)
		# env.env.stoch_step = True
		

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
				# print(f"return up to now is {env.ep_return}")
				ep_return.append(env.ep_return)
				state = next_state

		env.dump_json_result(os.path.join(results_folder, test_case, policy_name))
  
		avg_return+=env.ep_return


		# Close the env
		env.close()
	avg_return /= instance_count
	print(f"average return over {instance_count} instances:", avg_return)
	# average return over 20 instances: 3009.7017


if __name__ == "__main__":
	main()
