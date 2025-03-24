import gym
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from src.problems.road_charging.gym_env import RoadCharging

def policy(env, charge_lb=0.1, charge_ub=0.8,):
	actions = env.action_space.sample()
	# actions = env.rng.choice([0, 1], size=env.N)   

	future_free_chargers = 0
	for i in range(env.N):
		o_t_i, tau_t_i, SoC_i = env.evs.get_state(i) 
		if o_t_i == 2 and (tau_t_i == 0 or SoC_i >= charge_ub):
			future_free_chargers += 1  # Predict charger release based on SoC and status
			
		if tau_t_i >= 1:
			actions[i] = 0
		elif tau_t_i == 0 and SoC_i >= charge_ub:
			actions[i] = 0
		elif tau_t_i == 0 and SoC_i <= charge_lb:
			actions[i] = 1
   
	charging_candidates = [i for i, a in enumerate(actions) if a == 1]
   
	# Enforce charging station capacity limit
	# max_resource_limit = env.charging_stations.get_dynamic_resource_level()
	predicted_max_resource_limit = future_free_chargers + env.charging_stations.get_dynamic_resource_level()
	
	if len(charging_candidates) > predicted_max_resource_limit:
		# print("charging_candidates:", charging_candidates)
		# Randomly select max_resource_limit EVs to charge
		selected_charging = env.rng.choice(charging_candidates, predicted_max_resource_limit, replace=False)
		# print("selected_charging:", selected_charging)
		# Set only the selected EVs to charge, others revert to action 0
		actions = [1 if i in selected_charging else 0 for i in range(env.N)]
		
	return actions

def main():
	total_evs = 200
	total_chargers = 10
	resolution = 15
	start_hour = 0
	price_type = 1
	demand_type = 1
	SoC_type = 1
 
	# Define the params table
	params_table = {
		1: {5: (0.3, 0.4), 8: (0.3, 0.4), 10: (0.3, 0.8), 12: (0.3, 0.8), 15: (0.7, 0.8), 20: (0.7, 0.8),
			50: (0.3, 0.4), 80: (0.3, 0.4), 100: (0.3, 0.8), 120: (0.3, 0.8), 150: (0.7, 0.8), 200: (0.7, 0.8)},
		2: {50: (0.1, 0.2), 80: (0.1, 0.2), 100: (0.3, 0.4), 120: (0.7, 0.8), 150: (0.7, 0.95), 200: (0.9, 0.95)}
	}
	charge_lb_val, charge_ub_val = params_table[SoC_type][total_evs]
 
	num_test_instance = 20
	policy_name = "mod_random_prophet"
 
	debug = False
 
	# Define paths
	input_path = "input"
	type_path = f"price{price_type}_demand{demand_type}_SoC{SoC_type}_{total_chargers}for{total_evs}_{start_hour}to24_{resolution}min"
 
	results = []

	step_complete_rates = []
	step_trip_times = []

	for i in range(1, num_test_instance + 1):
		output_path = os.path.join("output", type_path, policy_name, f"instance{i}")
		eval_config_fname = os.path.join("input", type_path, f"instance{i}", 
										 f"eval_config{i}.json")
		
		os.makedirs(output_path, exist_ok=True)
		
		env = RoadCharging(eval_config_fname)
		
		# Run a single episode (expand if needed)
		for ep in range(1):  
			env.reset()

			if debug:
				# print(f"Instance {i}, Episode {ep}")
				print(f"--- Timepoint {env.current_timepoint} ---")
				print("Initial SoCs:", env.evs.init_SoCs)
				current_index = env.current_timepoint * env.dt // 30
				# Print charging prices at the current time point
				for k in env.charging_stations.stations.keys():
					print(f"Charging Price at Step {env.current_timepoint} for Station {k}:",
						env.charging_stations.stations[k]["real_time_prices"][current_index])

				# Print trip requests raised at the current time point
				current_trip_requests = {trip_id: details for trip_id, details in env.trip_requests.trip_queue.items()
										if details["raised_time"] == env.current_timepoint}

				print(f"Trip Requests at Step {env.current_timepoint}:")
				print(current_trip_requests)

				print("-" * 40)

			for _ in range(env.T):
				# Get current state of taxi 0
				
				# Sample an action from the action space
				# actions = env.action_space.sample()
				actions = policy(env, charge_lb=charge_lb_val, charge_ub=charge_ub_val,)
						
				action = actions[0]
				
				if debug:
					# o_t_i, tau_t_i, SoC_i = env.evs.get_state(0)
					# # Print the current timepoint and state information
					# print(f"--- Timepoint {env.current_timepoint} ---")
					# print(f"State: o_t = {o_t_i}, tau_t = {tau_t_i}, SoC_t = {SoC_i:.4f}")
					# print(f"Action taken: a_t = {action} (EV 1: a_t = {actions[1]}, EV 2: a_t = {actions[2]})")
					status = []
					SoCs = []
					tau = []
					for i in range(env.N):
						o_t_i, tau_t_i, SoC_i = env.evs.get_state(i)
						status.append(o_t_i)
						tau.append(tau_t_i)
						SoCs.append(SoC_i)
					print("Status:", status)
					print("tau:", tau)
					print("SoCs:", SoCs)
					print("Actions:", actions)
	

				# Take a simulation step
				_, _, done, info = env.step(actions)
	
				if debug:
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
 
		# Store episode metrics
		# ep_returns_list.append(env.ep_returns)
		# charging_cost_list.append(env.charging_cost)
		# added_soc_list.append(env.added_soc)
		# fare_earned_list.append(env.fare_earned)
		# complete_rate_list.append(env.complete_rate)

		# driver_earnings_list.append(env.driver_earnings)
		# driver_trip_time_list.append(env.driver_trip_time)
		# driver_idle_time_list.append(env.driver_idle_time)

		step_complete_rates.append(env.step_complete_rate)
		step_trip_times.append(env.step_trip_time)

		# visualize_trajectory(env.agents_trajectory)
		
		serializable_data = {key: value.tolist() for key, value in env.agents_trajectory.items()}
		if not debug:		
			with open(os.path.join(output_path, "agents_trajectory.json"), "w") as f:
						json.dump(serializable_data, f, indent=4)
		
		ep_results = env.print_ep_results()
		# print(json.dumps(ep_results, indent=4))
		if not debug:
			with open(os.path.join(output_path, "simulation_results.json"), "w") as f:
				json.dump(ep_results, f, indent=4)

		results.append(ep_results)

		if debug:
			print("Each driver's accumulated earnings:", env.driver_earnings)
			print("Total accumulated earnings:", sum(env.driver_earnings))
			print("Each driver's idle time due to rejection (competition, or high-SOC charging):",env.driver_idle_time)
			print("Total idle time:", sum(env.driver_idle_time))
			# print("Each driver's accumulated time to next decision:", env.time_to_next_decision)
			# print("Total time to next decision:", sum(env.time_to_next_decision))

		env.close()

	# After running all test instances
	# Initialize dictionaries for averaging scalars and lists
	avg_result = {key: 0 for key, v in results[0].items() if not isinstance(v, list)}
	list_result = {key: [] for key, v in results[0].items() if isinstance(v, list)}

	# Accumulate values
	for res in results:
		for k, v in res.items():
			if isinstance(v, list):  
				list_result[k].append(v)  # Store lists separately
			else:  
				avg_result[k] += v  # Sum scalar values

	num_instances = len(results)

	# Compute average for scalar values
	for k in avg_result:
		avg_result[k] /= num_instances

	# Compute element-wise average for list values
	for k in list_result:
		avg_result[k] = np.mean(list_result[k], axis=0).tolist()  # Convert back to list

	# Now avg_result contains both averaged scalars and averaged lists

	# print(avg_result)
	if not debug:
		with open(os.path.join("output", type_path, policy_name, "avg_result.json"), "w") as f:
				json.dump(avg_result, f, indent=4)
	
	
	# # Compute average statistics
	# avg_ep_returns = np.mean(ep_returns_list)
	# avg_charging_cost = np.mean(charging_cost_list)
	# avg_added_soc = np.mean(added_soc_list)
	# avg_fare_earned = np.mean(fare_earned_list)
	# avg_complete_rate = np.mean(complete_rate_list)

	# avg_driver_earnings = np.mean(driver_earnings_list, axis=0)
	# avg_driver_trip_time = np.mean(driver_trip_time_list, axis=0)
	# avg_driver_idle_time = np.mean(driver_idle_time_list, axis=0)

	avg_step_complete_rate = np.mean(step_complete_rates, axis=0)
	avg_step_trip_time = np.mean(step_trip_times, axis=0)

	# Print averaged results
	# print(f"Avg Ep Returns: {avg_ep_returns}, Avg Charging Cost: {avg_charging_cost}, Avg Added SoC: {avg_added_soc},")
	# print(f"Avg Fare Earned: {avg_fare_earned}, Avg Complete Rate: {avg_complete_rate}")

	# print(f"Avg Driver Earnings: {avg_driver_earnings}, Sum: {sum(avg_driver_earnings)}")
	# print(f"Avg Driver Trip Time: {avg_driver_trip_time}, Sum: {sum(avg_driver_trip_time)}")
	# print(f"Avg Driver Idle Time: {avg_driver_idle_time}, Sum: {sum(avg_driver_idle_time)}")

	# Plot averaged step metrics
	fig, ax1 = plt.subplots(figsize=(10,6))

	# ax1.plot(avg_step_complete_rate, label="Avg Step Complete Rate", color="tab:blue")
	ax1.step(range(len(avg_step_complete_rate)), avg_step_complete_rate, 
	     label="Avg Step Complete Rate", color="tab:orange", alpha=0.5, linewidth=1.5, where="post")
	# ax1.plot(avg_step_complete_rate, label="Avg Step Complete Rate", color="tab:orange", alpha=0.5, linewidth=1, )
	ax1.set_xlabel("Time Steps")
	ax1.set_ylabel("Complete Rate", color="tab:orange")
	ax1.tick_params(axis="y", labelcolor="tab:orange")

	ax2 = ax1.twinx()
	# ax2.plot(avg_step_trip_time, label="Avg Step Trip Time", color="tab:orange")
	ax2.step(range(len(avg_step_trip_time)), avg_step_trip_time, linestyle="dashed", linewidth=1.5, label="Avg Step Trip Time", color="tab:blue", where="post")
	ax2.set_ylabel("Trip Time", color="tab:blue")
	ax2.tick_params(axis="y", labelcolor="tab:blue")

	plt.title(f"Averaged Step Metrics Over {num_test_instance} Instances")
	fig.tight_layout()
	plt.savefig(os.path.join("output", type_path, policy_name, "avg_step_metrics.png"), dpi=300)
	plt.show()


   
   
if __name__ == "__main__":
	main()