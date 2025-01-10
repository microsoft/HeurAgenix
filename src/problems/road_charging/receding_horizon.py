from gurobipy import Model, GRB
import numpy as np
import time
import json
import csv
import os

# Explanation of receding horizon strategy:
# If the lookahead step is 2, at time 0, we solve for actions a0 and a1, and predict states s1 and s2.
# With a solution step of 1, we only apply the first action (a0) at time 0.
# At the next step (time 1), we solve again to obtain actions a1 and a2, and apply only a1.
# This process repeats, continuously updating the solution based on the latest state.

def rh_solver(config, start_time, lookahead_steps,
			   solution_steps, initial_states):
	
	M = 1000  # A large constant for Big-M formulation
	epsilon = 0.00001
	penalty_cost = -100
	
	n_EVs = config['fleet_size'] 
	n_chargers = config['n_chargers']  
	h = config['connection_fee($)']  
	d_rates = config['d_rates(%)']  
	c_rates = config['c_rates(%)']  
	low_SoC = config["low_SoC"]
	max_cap = config['max_cap'] 
	w = config["w"]
	p = config["p"]
	rt_samples = config["ride_data_instance"]
	max_rt_sample = max(max(sublist) for sublist in rt_samples)
	
	if start_time + lookahead_steps > max_time_step:
		T = max_time_step - start_time
	else:
		T = lookahead_steps
  
	# Initialize model
	model = Model("MultiAgentMDP")
	model.setParam('OutputFlag', 0) # disable output
	 
	# Set Gurobi parameters
	model.Params.MIPGap = 0.01  # Allow 1% optimality gap
	model.Params.TimeLimit = 300  # Set a 5-minute time limit
	model.Params.Presolve = 2  # Use aggressive presolve
	model.Params.Cuts = 2  # Enable aggressive cuts
	model.Params.Threads = 4  # Limit to 4 threads (if on shared server)
   
	# Variables
	a = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="a")  # Action
	z_order = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="take_order")  # Indicator
	z_connect = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="start_charge")  # Indicator
	z_charge = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="cont_charge")  # Indicator
	z_ride = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="on_ride")  # On a ride
	rt = model.addVars(n_EVs, T+1, lb=0, ub=max_rt_sample, vtype=GRB.INTEGER, name="rt")  # Ride time
	ct = model.addVars(n_EVs, T+1, vtype=GRB.BINARY, name="ct")  # Charging time
	SoC = model.addVars(n_EVs, T+1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="SoC")  # State of Charge
	assigned_rt = model.addVars(n_EVs, T, lb=0, ub=max_rt_sample, vtype=GRB.INTEGER, name="rt")  # Ride time
	
	clipped_cr = model.addVars(n_EVs, T, lb=0.0, ub=c_rates[0], vtype=GRB.CONTINUOUS, name="clipped c rate")  # State of Charge
	clipped_dr = model.addVars(n_EVs, T, lb=0.0, ub=d_rates[0], vtype=GRB.CONTINUOUS, name="clipped d rate")
	actual_cr = model.addVars(n_EVs, T, lb=0.0, ub=c_rates[0], vtype=GRB.CONTINUOUS, name="actual c rate")  # State of Charge
	actual_dr = model.addVars(n_EVs, T, lb=0.0, ub=d_rates[0], vtype=GRB.CONTINUOUS, name="actual d rate")
	
	z_okSoC = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="SoC above lowSoC")  # if above low_SoC, then z_okSoC=1
	z_nearFull = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="SoC near full")  # if 1-SoC < c_rate, then z_nearFull = 1
	z_nearEmpty = model.addVars(n_EVs, T, vtype=GRB.BINARY, name="SoC near empty")  # if SoC < d_rate, then z_nearEmpty = 1

	lower_bounds = {t: 0 for t in range(T)}
	penalty_violation = model.addVars(T, lb=lower_bounds, vtype=GRB.CONTINUOUS, name="penalty_violation")
 
	for i in range(n_EVs):
		model.addConstr(rt[i, 0] == initial_states['RideTime'][i], "Initial rt for agent %s at t=0" % i)
		model.addConstr(ct[i, 0] == initial_states['ChargingStatus'][i], "Initial ct for agent %s at t=0" % i)
		model.addConstr(SoC[i, 0] == initial_states['SoC'][i], "Initial SoC for agent %s at t=0" % i)

	for t in range(T): # 0, 1, ..., T-1
		for i in range(n_EVs):
		
			# Transition dynamics for 'rt'
			model.addGenConstrIndicator(z_ride[i, t], True, rt[i, t+1] == rt[i, t] - 1)
			model.addGenConstrIndicator(z_order[i, t], True, rt[i, t+1] == assigned_rt[i, t])
			model.addGenConstrIndicator(z_connect[i, t], True, rt[i, t+1] ==0)
			model.addGenConstrIndicator(z_charge[i, t], True, rt[i, t+1] == 0)
		
			model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] == rt_samples[i][start_time+t]*z_okSoC[i,t])
			model.addGenConstrIndicator(z_order[i, t], True, SoC[i, t] >= d_rates[i] * assigned_rt[i, t])		
			model.addGenConstrIndicator(z_order[i, t], False, assigned_rt[i, t] == 0)
			
			model.addConstr(SoC[i, t] >= low_SoC - M * (1 - z_okSoC[i,t]) )
			model.addConstr(SoC[i, t] <= low_SoC + M * z_okSoC[ i,t] - epsilon)

			model.addConstr(1-SoC[i, t] >= c_rates[i] - M * z_nearFull[i,t] )
			model.addConstr(1-SoC[i, t] <= c_rates[i] + M * (1-z_nearFull[i,t]) - epsilon)

			model.addConstr(SoC[i, t] >= d_rates[i] - M * z_nearEmpty[i,t] )
			model.addConstr(SoC[i, t] <= d_rates[i] + M * (1-z_nearEmpty[i,t]) - epsilon)

			model.addGenConstrIndicator(z_nearFull[i,t], True, clipped_cr[i, t] == 1-SoC[i,t])
			model.addGenConstrIndicator(z_nearFull[i,t], False, clipped_cr[i, t] == c_rates[i])
			
			model.addGenConstrIndicator(ct[i, t], True, actual_cr[i, t] == clipped_cr[i, t])
			model.addGenConstrIndicator(ct[i, t], False, actual_cr[i, t] == 0.0)

			model.addGenConstrIndicator(z_nearEmpty[i,t], True, clipped_dr[i, t] == SoC[i,t])
			model.addGenConstrIndicator(z_nearEmpty[i,t], False, clipped_dr[i, t] == d_rates[i])

			model.addConstr(z_nearEmpty[i,t] + z_nearFull[i,t] <= 1)

			model.addGenConstrIndicator(ct[i, t], True, actual_dr[i, t] == 0.0)
			model.addGenConstrIndicator(ct[i, t], False, actual_dr[i, t] == clipped_dr[i, t])

			# Transition dynamics for 'ct'
			model.addConstr(ct[i, t+1] == a[i, t]) 

			# SoC transition constraint
			model.addConstr(SoC[i, t+1] == SoC[i, t] + actual_cr[i,t] - actual_dr[i,t])
			
			# Constraints for indicator variables
			model.addConstr(z_ride[i, t] == 1 - z_order[i, t] - z_connect[i, t] - z_charge[i, t])
			model.addConstr(z_order[i, t] + z_connect[i, t] + z_charge[i, t] <= 1)

			# Constraints for z_connect (first-time connection to a charger)
			model.addConstr(z_connect[i, t] <= a[i, t])
			model.addConstr(z_connect[i, t] <= 1 - ct[i, t])
			model.addConstr(z_connect[i, t] >= a[i, t] - ct[i, t])

			# Constraints for z_charge (continue charging)
			model.addConstr(z_charge[i, t] <= a[i, t])
			model.addConstr(z_charge[i, t] <= ct[i, t])
			model.addConstr(z_charge[i, t] >= a[i, t] + ct[i, t] - 1)

			# Big-M constraints for z_order 
			model.addConstr(rt[i, t] + a[i, t] - 1 <= M * (1 - z_order[i, t]))
			model.addConstr(rt[i, t] + M * a[i, t] - 1 >= 1 - M * z_order[i, t])
			model.addConstr(z_order[i, t] <= 1 - a[i, t])
			
			# big-M for ride time and action: if ride time >= 2, action cannot be 1
			model.addConstr(rt[i, t]- 1 <= M * (1 - a[i, t]))

			# big-M for ride time and charging status: if ride time >=1, charging status cannot be 1
			model.addConstr(rt[i, t] <= M * (1 - ct[i, t]))
			
		# model.addConstr(
		#     sum(z_connect[i, t] for i in range(n_EVs)) + 
		#     sum(z_charge[i, t] for i in range(n_EVs)) <= n_chargers
		# )
  
	for t in range(T):
		# Global constraint for chargers
		model.addConstr(
			sum(a[i, t] for i in range(n_EVs)) - n_chargers - epsilon <= penalty_violation[t]
		)

	# end double loop

	# for t in range(T):
	# 	# Global constraint for chargers
	# 	model.addConstr(
	# 		sum(a[i, t] for i in range(n_EVs)) <= n_chargers + epsilon
	# 	)

	# Objective function
	objective = 0
	
	for t in range(T):  # 0, 1, ..., T-1
		for i in range(n_EVs):
			objective += w[start_time + t] * assigned_rt[i, t] 
			objective += - h * z_connect[i, t] - p[start_time + t] * actual_cr[i, t] * max_cap
		objective += penalty_cost * penalty_violation[t]
  
	model.setObjective(objective, GRB.MAXIMIZE)

	# Optimize
	model.optimize()

	# Check optimization status
	if model.status != GRB.OPTIMAL:
		print(f"Optimization failed with status {model.status}")
		return {}, {}

	rt_values = [[] for _ in range(n_EVs)]
	ct_values = [[] for _ in range(n_EVs)]
	SoC_values = [[] for _ in range(n_EVs)]
	assigned_rt_values = [[] for _ in range(n_EVs)]
	a_values = [[] for _ in range(n_EVs)]
	actual_c_rates = [[] for _ in range(n_EVs)]
	actual_d_rates = [[] for _ in range(n_EVs)]

	for i in range(n_EVs):
		for t in range(solution_steps):
			rt_values[i].append(rt[i, t].X)
			ct_values[i].append(ct[i, t].X)
			SoC_values[i].append(SoC[i, t].X)
			a_values[i].append(a[i, t].X)
			assigned_rt_values[i].append(assigned_rt[i, t].X)
			actual_c_rates[i].append(actual_cr[i,t].X)
			actual_d_rates[i].append(actual_dr[i,t].X)

	current_solution = { 
		"RideTime": rt_values,
		"ChargingStatus": ct_values,
		"SoC": SoC_values,
		"assigned_RideTime": assigned_rt_values,
		"actions": a_values,
		"actual c rates": actual_c_rates,
		"actual d rates": actual_d_rates
	}

	next_initial_state = {
            # Ensure RideTime values are integers
            "RideTime": [int(round(rt[i, solution_steps].X)) for i in range(n_EVs)],  
            
            # Ensure ChargingStatus values are binary
            "ChargingStatus": [
                int(round(ct[i, solution_steps].X)) for i in range(n_EVs)
            ],  
            
            # Ensure SoC values are within [0, 1]
            "SoC": [
                max(0.0, min(1.0, SoC[i, solution_steps].X)) for i in range(n_EVs)
            ],  
    }

	recovered_obj_val = 0
	for i in range(n_EVs):
		for t in range(solution_steps):
			recovered_obj_val += w[start_time + t] * assigned_rt[i, t].X - (h * z_connect[i, t].X + p[start_time + t] * actual_cr[i, t].X * max_cap)

	penalty_for_violations = 0
	for t in range(solution_steps):  # 0, 1, ..., T-1
		penalty_for_violations += penalty_cost * penalty_violation[t].X
		
	return current_solution, next_initial_state, recovered_obj_val, penalty_for_violations


if __name__ == "__main__":

	n_EVs = 10
	test_case = f"all_days_negativePrices_polarizedInitSoC_1for{n_EVs}"
  lookahead_steps = 16
	solution_steps = 1
	
	avg_runtime = 0
	avg_return = 0
	instance_count = 1
  data_folder = "test_cases_adjusted"
	results_folder = "results_updated"
	policy_name = f"rh_lookahead_{lookahead_steps}steps"
	
	for instance_num in range(1, 1+instance_count):
	
		data_file = os.path.join(data_folder, test_case, f"config{instance_num}_{n_EVs}EVs_1chargers.json")
		
		with open(data_file, 'r') as json_file:
			config = json.load(json_file)
			
		initial_SoCs = config["initial_SoCs"]
		max_time_step = config["max_time_step"]
  
		start_time = 0
		final_returns = 0
		total_penalty = 0
		solution_history = []
		initial_states = {
			'RideTime': [0] * n_EVs,
			'ChargingStatus': [0] * n_EVs,
			'SoC': initial_SoCs
		}
		
		start_runtime = time.time()  # Record the start time
		while start_time < max_time_step:
			try:
				current_solution, next_initial_state, obj_val, penalty_val = rh_solver(config, 
																		start_time,
																			lookahead_steps,
																			solution_steps,
																			initial_states)
			except Exception as e:
				print(f"Solver failed at step {start_time}: {str(e)}")
				break
	
			solution_history.append(current_solution)
			initial_states = next_initial_state
			final_returns += obj_val
			total_penalty += penalty_val
			
			start_time += solution_steps
			
			if start_time % 10 == 0:
				print(f"Instance: {instance_num}, Start Time: {start_time}, Objective Value: {obj_val:.2f}")
		
		end_runtime = time.time()
			
		final_solution = {
			"RideTime": [[] for _ in range(n_EVs)],  
			"ChargingStatus": [[] for _ in range(n_EVs)],  
			"SoC": [[] for _ in range(n_EVs)],  
			"assigned_RideTime": [[] for _ in range(n_EVs)], 
			"actions": [[] for _ in range(n_EVs)] 
		}

		for solution in solution_history:
			for i in range(n_EVs):  
				final_solution["RideTime"][i].extend(solution["RideTime"][i])  
				final_solution["ChargingStatus"][i].extend(solution["ChargingStatus"][i])  
				final_solution["SoC"][i].extend(solution["SoC"][i])  
				final_solution["assigned_RideTime"][i].extend(solution["assigned_RideTime"][i])  
				final_solution["actions"][i].extend(solution["actions"][i])  
				
		for i in range(n_EVs):
			final_solution["RideTime"][i].append(next_initial_state['RideTime'][i])
			final_solution["SoC"][i].append(next_initial_state['SoC'][i])  
			final_solution["ChargingStatus"][i].append(next_initial_state['ChargingStatus'][i])  
			
		final_solution["Final Returns"] = final_returns
		final_solution["Total Penalty"] = total_penalty
		final_solution["runtime"] = end_runtime - start_runtime
		
		save_dir = os.path.join(results_folder, test_case, f"rh_penalty_H{lookahead_steps}_debug")
		os.makedirs(save_dir, exist_ok=True)	
		with open(os.path.join(save_dir, f"instance{instance_num}_solutoin.json"), "w") as f:
			json.dump(final_solution, f, indent=4) 
			
		avg_runtime += final_solution["runtime"]
		avg_return += final_solution["Final Returns"] 
		
	avg_runtime /= instance_count
	avg_return /= instance_count

	print(f"Average results over {instance_count} instances:")
	print(f"Average Runtime: {avg_runtime:.2f} seconds")
	print(f"Average Final Return: {avg_return:.2f}")
			
