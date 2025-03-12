import gym
from gym import spaces
import numpy as np
import json
import os
from utils import visualize_trajectory, get_data_type_meanings
from gym_env import RoadCharging



def policy(env):
	actions = env.action_space.sample()

	return actions


def main():
	# example_config = {
	# 	"total_time_steps": 216,
	# 	"time_step_minutes":5,
	# 	"total_evs": 5,
	# 	"committed_charging_block_minutes": 15,
	# 	"renewed_charging_block_minutes": 5, 
	# 	"ev_params": None,
	# 	"trip_params": None,
	# 	"charging_params": None,
	# 	"other_env_params": None
	# }
	total_evs = 3
	total_chargers = 1
	resolution = 15
	start_hour = 6

	price_type = 1
	demand_type = 1
	SoC_type = 1
 
	test_instance_num=1
	policy_name = "random"

	# Define paths
	input_path = "input"
	type_path = f"price{price_type}_demand{demand_type}_SoC{SoC_type}_{total_chargers}for{total_evs}_{start_hour}to24_{resolution}min"
 
	results = []

	for i in range(1, test_instance_num + 1):
		output_path = os.path.join("output", type_path, policy_name, f"instance{i}")
		eval_config_fname = os.path.join("input", type_path, f"instance{i}", 
										 f"eval_config{i}.json")
		
		os.makedirs(output_path, exist_ok=True)
		
		env = EVChargingEnv(eval_config_fname)
		
		# Run a single episode (expand if needed)
		for ep in range(1):  
			env.reset()
			print(f"Instance {i}, Episode {ep}")
			print("Initial SoCs:", env.evs.init_SoCs)
			for k in env.charging_stations.stations.keys():
				print("Charging Prices (first 5):", env.charging_stations.stations[k]["real_time_prices"][:5])
			print("Trip Requests (first 5):")
			first_5_trip_requests = dict(list(env.trip_requests.trip_queue.items())[:5])
			print(first_5_trip_requests)
			print("-" * 40)

			for _ in range(env.T):
				# Get current state of taxi 0
				o_t_i, tau_t_i, SoC_i = env.evs.get_state(0)
				# Sample an action from the action space
				# actions = env.action_space.sample()
				actions = policy(env)
				action = actions[0]
				
				# Print the current timepoint and state information
				print(f"--- Timepoint {env.current_timepoint} ---")
				print(f"State: o_t = {o_t_i}, tau_t = {tau_t_i}, SoC_t = {SoC_i:.4f}")
				print(f"Action taken: a_t = {action} (EV 1: a_t = {actions[1]}, EV 2: a_t = {actions[2]})")
				
				# Take a simulation step
				_, _, done, info = env.step(actions)
	
				op_status = env.states["OperationalStatus"]
				idle_evs = [i for i, status in enumerate(op_status) if status == 0]
				serving_evs = [i for i, status in enumerate(op_status) if status == 1]
				charging_evs = [i for i, status in enumerate(op_status) if status == 2]
				total_available_chargers = env.charging_stations.get_dynamic_resource_level()
				open_stations = env.charging_stations.update_open_stations()
				newly_requested_trips = [(key,value['raised_time'],value['trip_duration'],value['status']) for key, value in env.trip_requests.trip_queue.items() 
						if value['raised_time'] == env.current_timepoint-1]

				print(f"After step(): Idle EVs: {idle_evs}, Serving EVs: {serving_evs}, Charging EVs: {charging_evs}, "
				f"Available Chargers: {total_available_chargers}, Open Stations: {open_stations}")
				print(f"Newly requested trips: {newly_requested_trips}")

				# Interpret the action
				if action == 0:
					print("Dispatch order:")
				else:
					print("Relocate to charge:")
				
				# Check for charging session info
				cs_result = env.dispatch_results[0].get("cs")
				if cs_result:
					print(f"  Session added SoC: {cs_result.get('session_added_SoC'):.4f}")
				
				# Check for order (trip) info
				order_result = env.dispatch_results[0].get("order")
				if order_result:
					print(f"  Trip duration: {order_result.get('trip_duration')} minutes, "
						f"Trip fare: {order_result.get('trip_fare')}")
				
				print("=" * 40)

				
				# if ep % 5 == 0:
				# 	env.report_progress()
		
				if done:
					break

		visualize_trajectory(env.agents_trajectory)
	
		serializable_data = {key: value.tolist() for key, value in env.agents_trajectory.items()}
		with open(os.path.join(output_path, "agents_trajectory.json"), "w") as f:
			json.dump(serializable_data, f, indent=4)
		
		ep_results = env.print_ep_results()
		print(json.dumps(ep_results, indent=4))
		with open(os.path.join(output_path, "simulation_results.json"), "w") as f:
			json.dump(ep_results, f, indent=4)
   
		results.append(ep_results)
		env.close()

	# After running all test instances
	avg_result = {key: 0 for key in results[0].keys()}

	for res in results:
		for k, v in res.items():
			avg_result[k] += v

	num_instances = len(results)
	for k, v in avg_result.items():
		avg_result[k] = v / num_instances

	print(avg_result)
   
   
if __name__ == "__main__":
	main()