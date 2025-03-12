import gurobipy as gp
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import csv
import os
from feasibility_check import feasibility_check


def main():
	
	n_EVs = 5
	n_chargers = 1
	SoC_data_type = "high"
	test_case = f"all_days_negativePrices_{SoC_data_type}InitSoC_{n_chargers}for{n_EVs}"
	data_folder = "test_cases_adjusted"
	results_folder = "results_updated"
	policy_name = "milp_solver_5min"
	
	M = 1000      # A large constant for Big-M formulation
	epsilon = 0.00001
	penalty_cost = -100

	avg_runtime = 0
	avg_return = 0
	avg_violation = 0
	instance_count = 20
	
	for instance_num in range(1, 1+instance_count):
	 
		data_file = os.path.join(data_folder, test_case, f"config{instance_num}_{n_EVs}EVs_1chargers.json")
		
		with open(data_file, 'r') as json_file:
			config = json.load(json_file)
		
		T = config["max_time_step"] 
		n_agents = config['fleet_size']  # Number of EVs in the fleet
		n_chargers = config['n_chargers']  # Total number of chargers
		h = config['connection_fee($)']  # $ per connection session
		d_rates = config['d_rates(%)']  # Battery consumption rate per time step
		c_rates = config['c_rates(%)']   # Charger speed per time step
		initial_SoCs = config["initial_SoCs"]
		low_SoC = config["low_SoC"]
		max_cap = config['max_cap'] 
		w = config["w"]
		p = config["p"]
		rt_samples = config["ride_data_instance"]
		max_rt_sample = max(max(sublist) for sublist in rt_samples)
		# print("rt/assigned rt upper bound:", max_rt_sample)

		# Initialize model
		model = Model("MultiAgentMDP")
		model.Params.MIPGap = 0.01  # Allow 1% optimality gap
		model.Params.TimeLimit = 300  # Set a 5-minute time limit
		model.Params.Presolve = 2  # Use aggressive presolve
		model.Params.Cuts = 2  # Enable aggressive cuts
		model.Params.Threads = 4  # Limit to 4 threads (if on shared server)

		# Variables
		a = model.addVars(n_agents, T, vtype=GRB.BINARY, name="a")  # Action
		z_order = model.addVars(n_agents, T, vtype=GRB.BINARY, name="take_order")  # Indicator
		z_connect = model.addVars(n_agents, T, vtype=GRB.BINARY, name="start_charge")  # Indicator
		z_charge = model.addVars(n_agents, T, vtype=GRB.BINARY, name="cont_charge")  # Indicator
		z_ride = model.addVars(n_agents, T, vtype=GRB.BINARY, name="on_ride")  # On a ride
		rt = model.addVars(n_agents, T+1, lb=0, ub=max_rt_sample, vtype=GRB.INTEGER, name="rt")  # Ride time
		ct = model.addVars(n_agents, T+1, vtype=GRB.BINARY, name="ct")  # Charging time
		SoC = model.addVars(n_agents, T+1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="SoC")  # State of Charge
		assigned_rt = model.addVars(n_agents, T, lb=0, ub=max_rt_sample, vtype=GRB.INTEGER, name="rt")  # Ride time
		
		clipped_cr = model.addVars(n_agents, T, lb=0.0, ub=c_rates[0], vtype=GRB.CONTINUOUS, name="clipped c rate")  # State of Charge
		clipped_dr = model.addVars(n_agents, T, lb=0.0, ub=d_rates[0], vtype=GRB.CONTINUOUS, name="clipped d rate")
		actual_cr = model.addVars(n_agents, T, lb=0.0, ub=c_rates[0], vtype=GRB.CONTINUOUS, name="actual c rate")  # State of Charge
		actual_dr = model.addVars(n_agents, T, lb=0.0, ub=d_rates[0], vtype=GRB.CONTINUOUS, name="actual d rate")
		
		z_okSoC = model.addVars(n_agents, T, vtype=GRB.BINARY, name="SoC above lowSoC")  # if above low_SoC, then z_okSoC=1
		z_nearFull = model.addVars(n_agents, T, vtype=GRB.BINARY, name="SoC near full")  # if 1-SoC < c_rate, then z_nearFull = 1
		z_nearEmpty = model.addVars(n_agents, T, vtype=GRB.BINARY, name="SoC near empty")  # if SoC < d_rate, then z_nearEmpty = 1

		lower_bounds = {t: 0 for t in range(T)}
		penalty_violation = model.addVars(T, lb=lower_bounds, vtype=GRB.CONTINUOUS, name="penalty_violation")

		for i in range(n_agents):
			
			model.addConstr(rt[i, 0] == 0, "Initial rt for agent %s at t=0" % i)
			model.addConstr(ct[i, 0] == 0, "Initial ct for agent %s at t=0" % i)
			model.addConstr(SoC[i, 0] == initial_SoCs[i], "Initial SoC for agent %s at t=0" % i)

		for t in range(T): # 0, 1, ..., T-1
			for i in range(n_agents):
			
				# Transition dynamics for 'rt'
				model.addGenConstrIndicator(z_ride[i, t], True, rt[i, t+1] == rt[i, t] - 1)
				model.addGenConstrIndicator(z_order[i, t], True, rt[i, t+1] == assigned_rt[i, t])
				model.addGenConstrIndicator(z_connect[i, t], True, rt[i, t+1] ==0)
				model.addGenConstrIndicator(z_charge[i, t], True, rt[i, t+1] == 0)
			
				model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] == rt_samples[i][t]*z_okSoC[i,t])
				model.addGenConstrIndicator(z_order[i, t], False, assigned_rt[i, t] == 0)
				
				model.addGenConstrIndicator(z_order[i, t], True, SoC[i, t] >= d_rates[i] * assigned_rt[i, t])
	
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
			#     sum(z_connect[i, t] for i in range(n_agents)) + 
			#     sum(z_charge[i, t] for i in range(n_agents)) <= n_chargers
			# )

		for t in range(T):
				# Global constraint for chargers
				model.addConstr(
					sum(a[i, t] for i in range(n_agents)) - n_chargers - epsilon <= penalty_violation[t]
				)

		objective = 0
		for t in range(T):  # 0, 1, ..., T-1
			for i in range(n_agents):

				objective += w[t] * assigned_rt[i, t] - (h * z_connect[i, t] + p[t] * actual_cr[i, t] * max_cap)

			objective += penalty_cost * penalty_violation[t]

		model.setObjective(objective, GRB.MAXIMIZE)

		# Optimize
		model.optimize()

		# Print the meaning of the status
		status = model.status
		if  model.status == GRB.OPTIMAL:
			print("The model is solved to optimality.")
			status = "optimal"
		elif  model.status == GRB.INFEASIBLE:
			print("The model is infeasible.")
			status = "infeasible"
		elif  model.status == GRB.UNBOUNDED:
			print("The model is unbounded.")
			status = "unbounded"
		elif  model.status == GRB.INF_OR_UNBD:
			print("The model is infeasible or unbounded.")
			status = "infeasible or unbounded"
		elif  model.status == GRB.TIME_LIMIT:
			print("Optimization stopped due to a time limit.")
			status = "reach time limit"
		elif  model.status == GRB.INTERRUPTED:
			print("Optimization was interrupted.")
			status = "interrupted"
		elif  model.status == GRB.NUMERIC:
			print("Optimization was terminated due to numerical issues.")
			status = "numerical issues"
		elif  model.status == GRB.SUBOPTIMAL:
			print("The model has a suboptimal solution.")
			status = "suboptimal"
		else:
			print(f"Status code {status}: Check Gurobi documentation for details.")

		with open("variable_values.txt", "w") as f:
				for var in model.getVars():
					f.write(f"{var.varName}: {var.x}\n")
			
		rt_values = [[] for _ in range(n_agents)]
		ct_values = [[] for _ in range(n_agents)]
		SoC_values = [[] for _ in range(n_agents)]
		assigned_rt_values = [[] for _ in range(n_agents)]
		a_values = [[] for _ in range(n_agents)]
		for i in range(n_agents):
			for t in range(T+1):
				rt_values[i].append(rt[i, t].X)
				ct_values[i].append(ct[i, t].X)
				SoC_values[i].append(SoC[i, t].X)

		for i in range(n_agents):
			for t in range(T):
				a_values[i].append(a[i, t].X)
				assigned_rt_values[i].append(assigned_rt[i, t].X)
				
		for i in range(n_agents):
			for t in range(T):
				if z_ride[i,t].X >= 1 and ct[i,t].X == 1:
					print("Wrong.")


		recovered_obj_val = 0
		order_payments = []
		charging_costs = []
		for i in range(n_agents):
			for t in range(T):
				if i == 0:
					order_payments.append(w[t] * assigned_rt[i, t].X)
					charging_costs.append(h * z_connect[i, t].X + p[t] * actual_cr[i, t].X * max_cap)
				recovered_obj_val += w[t] * assigned_rt[i, t].X - (h * z_connect[i, t].X + p[t] * actual_cr[i, t].X * max_cap)

		print("recovered objective value:", recovered_obj_val)

		penalty_for_violations = 0
		for t in range(T):  # 0, 1, ..., T-1
		
			penalty_for_violations += penalty_cost * penalty_violation[t].X
  
				
		time_limit_reached = model.status == gp.GRB.TIME_LIMIT  # Check if time limit was reached
		best_objective = model.objVal  # Best objective value found
		best_bound = model.objBound  # Best bound (the current bound)
		gap = model.MIPGap  # MIP gap, the relative gap between best bound and best objective
		runtime = model.runtime  # Time taken for optimization
		
		solver_info = {
			"time_limit_reached": time_limit_reached,  # True or False based on time limit status
			"best_objective": best_objective,  # The best objective found
			"best_bound": best_bound,  # The best bound at the end of optimization
			"gap": gap,  # The gap between the objective and bound
			"runtime": runtime  # Time taken for the optimization
		}
					
		gurobi_solution = {
			"fleet_size": n_agents,
			"max_time_steps": T,
			"RideTime": rt_values,
			"ChargingStatus": ct_values,
			"SoC": SoC_values,
			"assigned_RideTime": assigned_rt_values,
			"actions": a_values,
			"order_payments": order_payments,
			"charging_costs": charging_costs,
			"returns": recovered_obj_val,
   			"violations penalty": penalty_for_violations,
			"Number of constraints violated": round(penalty_for_violations/penalty_cost),
			"solver_status": status,
			"solver_info": solver_info,
		}
		
		feasibility_check(gurobi_solution, n_chargers)
		
		save_dir = os.path.join(results_folder, test_case, policy_name)
		os.makedirs(save_dir, exist_ok=True)	
		with open(os.path.join(save_dir, f"instance{instance_num}_solutoin.json"), "w") as f:
			json.dump(gurobi_solution, f, indent=4)  # Use indent for readability

		
		avg_runtime += runtime
		avg_return += recovered_obj_val
		avg_violation += penalty_for_violations
		
	avg_runtime /= instance_count
	avg_return /= instance_count
	avg_violation /= instance_count

	print(f"Average results over {instance_count} instances:")
	print(f"Average Runtime: {avg_runtime:.2f} seconds")
	print(f"Average Final Return: {avg_return:.2f}")
	print(f"Average Final Violations: {avg_violation:.2f}")


if __name__ == "__main__":
	main()
