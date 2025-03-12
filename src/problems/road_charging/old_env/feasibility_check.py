import json
import numpy as np
import os

def feasibility_check(solution, n_chargers):
	
	epsilon = 0.00001
	
	required_keys = {"actions", "RideTime", "ChargingStatus", "SoC"}
	missing_keys = required_keys - solution.keys()
	assert not missing_keys, f"Solution is missing required keys: {missing_keys}"
	
	actions = solution["actions"]
	ride_time = solution["RideTime"]
	charging_status = solution["ChargingStatus"]
	SoC = solution["SoC"]

	n_agents = len(actions)
	max_time_step = len(actions[0])
	
	assert len(ride_time) == n_agents and len(ride_time[0]) == max_time_step + 1, (
		"RideTime must have one more time step than actions."
	)
	assert len(charging_status) == n_agents and len(charging_status[0]) == max_time_step + 1, (
		"ChargingStatus must have one more time step than actions."
	)
	assert len(SoC) == n_agents and len(SoC[0]) == max_time_step + 1, (
		"SoC must have one more time step than actions."
	)
	
	# Check individual constraints for each agent and time step
	for t in range(max_time_step):
		for i in range(n_agents):
			state_t = (int(ride_time[i][t]), int(charging_status[i][t]), SoC[i][t])
			next_state = (int(ride_time[i][t+1]), int(charging_status[i][t+1]), SoC[i][t+1])
			action_t = int(actions[i][t])
			
			if state_t[0] >= 2:
				assert action_t == 0, (
				f"Agent {i} at time {t}: Action must be 0 if ride leading time >= 2."
			)
			
			if state_t[2] > 1.0:
				assert action_t == 0, (
					f"Agent {i} at time {t}: Continuing to charge would exceed battery capacity."
				)
	for t in range(max_time_step):
		a = np.array(actions)
		assert sum(a[:, t]) <= n_chargers + epsilon, \
				print(f"Total charging requests exceed "
					f"available chargers ({n_chargers}) at time step {t}.")

if __name__ == "__main__":

	n_EVs = 8
	test_case = f"all_days_negativePrices_polarizedInitSoC_1for{n_EVs}"
	data_folder = "test_cases_adjusted"
	results_folder = "results_updated"
	policy_name = "base_policy"
	
	
	instance_count = 3
	for instance_num in range(1, 1+instance_count):
     
		data_file = os.path.join(data_folder, test_case, f"config{instance_num}_{n_EVs}EVs_1chargers.json")
		# solution_file = os.path.join("results", test_case, policy_name, f"instance{instance_num}", f"instance{instance_num}_solutoin.json")  
		solution_file = os.path.join(results_folder, test_case, policy_name, f"instance{instance_num}_solutoin.json")  

		with open(data_file, 'r') as json_file:
			config = json.load(json_file)

		with open(solution_file, 'r') as json_file:
			solution = json.load(json_file)
		
		print(f"Checking feasibility of solution for instance {instance_num}...")
		feasibility_check(solution, config["n_chargers"])
		print("Pass!")

		
		
