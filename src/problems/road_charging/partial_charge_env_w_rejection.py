import gym
from gym import spaces
import numpy as np
import json
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
from utils import load_file, csv_to_list, visualize_trajectory
from ChargingStations import ChargingStations
from TripRequests import TripRequests
from EVFleet import EVFleet


class EVChargingEnv(gym.Env):
	def __init__(self, config_fname: str):
	
		config = load_file(config_fname)

		self.T = config.get("total_time_steps", 96)
		self.dt = config.get("time_step_minutes", 15)
		self.N = config.get("total_evs", 5)
		self.min_ct = config.get("committed_charging_block_minutes", 15)
		self.renew_ct = config.get("renewed_charging_block_minutes", 15)
		self.memory = deque(maxlen=10000)  # Store all transitions here
		self.start_hour = config.get("operation_start_hour", 6)
		
		self.evs = EVFleet(config["ev_params"])
		self.trip_requests = TripRequests(config["trip_params"])
		self.charging_stations = ChargingStations(config["charging_params"])
		
		self.other_env_params = config.get("env_params", {})

		# Action space: 2 actions (remain-idle, go-charge)
		self.action_space = spaces.MultiBinary(self.N)

		# State space: For each EV, (operational-status, time-to-next-availability, SoC, location)

		self.state_space = spaces.Dict({
			"OperationalStatus": spaces.MultiDiscrete([3] * self.N),  # 3 states: 0 (idle), 1 (serving), 2 (charging)
			"TimeToNextAvailability": spaces.MultiDiscrete([101] * self.N),  # Values from 0 to 100
			"SoC": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32) # SoC in range [0, 1]
		})


	def reset(self, stoch_step: bool=False):
	
		if stoch_step:
			self.rng = np.random
		else:
			self.rng = np.random.default_rng(42)
   
		self.stoch_step = stoch_step
  
		self.evs.reset(self.rng)
		self.trip_requests.reset(self.rng)
		self.charging_stations.reset(self.rng)
		
		self.current_timepoint = 0
		self.states = self.evs.get_all_states()
  
		self.trip_requests.customer_arrivals = [int(np.ceil(x/200*self.N)) for x in self.trip_requests.customer_arrivals]
		
		if stoch_step: # if stoch_step, resample init SoCs and real time prices for each episode
			self.evs.reset_init_SoCs()
			self.charging_stations.update_real_time_prices() 
   
		if stoch_step is False:
			self.trip_requests.load_saved_trip_data()

		self.dispatch_results = {i: {'order': None, 'cs': None} for i in range(self.N)}  # Initialize dispatch results
		self.agents_trajectory = {
			'OperationalStatus': np.zeros((self.N, self.T+1)),
			'TimeToNextAvailability': np.zeros((self.N, self.T+1)),
			'SoC': np.zeros((self.N, self.T+1)),
			'Action': np.zeros((self.N, self.T), dtype=int),
			'Reward': np.zeros((self.N, self.T))
		}

		# Reset metrics
		self.total_charging_cost = 0.0
		self.total_added_soc = 0.0
		self.total_trip_requests = 0
		self.total_successful_dispatches = 0
		self.total_trip_fare_earned = 0.0
		self.total_penalty = 0.0
		self.acceptance_rate = 0.0
		self.total_idle_time = [0.0]*self.N
		self.ep_returns = 0  # Accumulate total returns over T timesteps

		return np.array(self.states)

	def show_config(self):
		plt.figure()
		for k in self.charging_stations.stations.keys():
			plt.plot(self.charging_stations.stations[k]["real_time_prices"], marker='s')
			print("len of lmp_rt:", len(self.charging_stations.stations[k]["real_time_prices"]))
		plt.ylabel("electricity prices ($/kWh)")
		plt.show()
  
		plt.figure()
		plt.plot(self.trip_requests.customer_arrivals, marker='o')
		print("len of customer arrivals:", len(self.trip_requests.customer_arrivals))
		plt.ylabel(f"customer arrivals per {self.dt} minutes")
		plt.show()
  
		plt.figure()
		plt.plot(self.trip_requests.per_minute_rates, marker='x')
		print("len of per minute pay rates:", len(self.trip_requests.per_minute_rates))
		plt.ylabel("driver pay per minute ($)")
		plt.show()
  
	def check_action_feasibility(self, actions):
		"""Checks feasibility of charging actions and applies penalties for constraint violations."""

		# Initialize penalty dictionary for each EV and an overall system constraint
		penalty = {i: 0.0 for i in range(self.N)}
		penalty["max_resource_limit"] = 0.0

		for i in range(self.N):
			o_t_i, tau_t_i, SoC_i = self.evs.get_state(i)  # Get EV state
			# remaining_cap = max(self.evs.max_SoC - SoC_i, 0)

			if actions[i] == 1:  # Charging action
				if tau_t_i >= 1:
					penalty[i] += 100  # Penalize charging while already busy (serving or charging)
				# Since this case does not reject any action, we don't need to enforce the penalty.
				# A poor decision will already result in a low return.
					print("Warning: Charging while already busy (serving or charging).")

				else:
					SoC_i = self.evs.get_state(i)[2]
					remaining_cap = max(self.evs.max_SoC - SoC_i, 0)
					session_added_SoC = self.evs.charge_rate[i] * (
						self.renew_ct if o_t_i == 2 else self.min_ct
					)
					session_added_SoC = min(session_added_SoC, remaining_cap)

					# if remaining_cap < session_added_SoC:
					# 	penalty[i] += 100  # Penalize charging when SoC is already high
					actual_charging_time = session_added_SoC / self.evs.charge_rate[i]
					session_time = self.renew_ct if o_t_i == 2 else self.min_ct
					idle_time = session_time  - actual_charging_time
					idle_time = max(idle_time, 0)  # Ensure no negative idle time
					SoC_drop = self.evs.energy_consumption[i] *idle_time
		
					actual_added_SoC = session_added_SoC - SoC_drop 
					if actual_added_SoC <= 0:
						penalty[i] += 100
						print("Warning: Charging failed to increase SoC.")

		# Apply system-level penalty if charging demand exceeds available slots
		if sum(actions) > self.charging_stations.get_dynamic_resource_level():
			penalty["max_resource_limit"] += 1000
			print("Warning: Charging demand exceeds available slots.")

		return penalty


	def step(self, actions):
		penalty = self.check_action_feasibility(actions)
		step_penalty = sum(penalty.values())
		self.total_penalty += step_penalty

		# or raise exception:
		# if step_penalty > 0:
		# 	raise Exception("Takes charging action while serving or when SoC is already high or exceed max resource limit.")

		# Step 1: Take in new customer requests
		if self.stoch_step:
			num_requests = int(self.trip_requests.customer_arrivals[int(self.start_hour*(60/self.dt))+self.current_timepoint]) 
			self.trip_requests.sample_requests(num_requests, self.current_timepoint)

		for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
			self.agents_trajectory[s][:, self.current_timepoint] = self.states[s]

		self.agents_trajectory['Action'][:, self.current_timepoint] = actions

		dispatch_evs = []  
		go_charge_evs = []
		stay_charge_evs = []

		# Step 2: Decide which evs for dispatch, which for charge
		for i in range(self.N):
			o_t_i, tau_t_i, SoC_i = self.evs.get_state(i)

			# Case 1: Idle EVs taking action 0 (remain idle)
			if tau_t_i == 0 and actions[i] == 0:
	
				dispatch_evs.append(i)
	 
				# If it was previously serving, remove its trip record
				if o_t_i == 1:
					self.dispatch_results[i]['order'] = None
					assert self.dispatch_results[i]['cs'] == None, "to be dispatch: previous charge records should be none"

				# If it was previously charging, release its charger slot
				if o_t_i == 2:
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_stations.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
					assert self.dispatch_results[i]['order'] == None, "to be dispatch: previous serving records should be none"
				
			# Case 2: Idle EVs taking action 1 (go charge)
			if tau_t_i == 0 and actions[i] == 1:
				if o_t_i == 0 or o_t_i == 1:
					go_charge_evs.append(i)
	
					# in case it is serving before, remove its trip records
					self.dispatch_results[i]['order'] = None

				if o_t_i == 2:
					stay_charge_evs.append(i)
					assert self.dispatch_results[i]['order'] == None, "to relocate to charge: previous serving records should be none"

	  
			if tau_t_i >= 1:
				continue
		
		# Step 3: Dispatch or relocate to charge
		if dispatch_evs:
			self.random_dispatch(dispatch_evs)
   
		if stay_charge_evs:
			for i in stay_charge_evs:
				SoC_i = self.evs.get_state(i)[2]
				remaining_cap = max(self.evs.max_SoC - SoC_i, 0)
				session_added_SoC = self.evs.charge_rate[i] * self.renew_ct
				session_added_SoC = min(session_added_SoC, remaining_cap)

				# Check if EV has an active charging session
				if i not in self.dispatch_results or self.dispatch_results[i]['cs'] is None:
					print(f"Warning: EV {i} has no active charging session.")
					continue  # Skip this EV

				charging_session = self.dispatch_results[i]['cs']
				
				actual_charging_time = session_added_SoC / self.evs.charge_rate[i]
				idle_time = self.renew_ct - actual_charging_time
				idle_time = max(idle_time, 0)  # Ensure no negative idle time
				SoC_drop = self.evs.energy_consumption[i] *idle_time
				# actual_added_SoC = max(session_added_SoC - SoC_drop, 0)  # Prevent negative SoC gain
				# session added SoC can be 0 if remaining SoC is 0
				# then actual_added_SoC is negative. should not apply max(,0)
				actual_added_SoC = session_added_SoC - SoC_drop
	
				if actual_added_SoC <= 0:
					# reject charging action, release charger
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_stations.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
					continue

				# Renew charging session
				charging_price = charging_session.get("price")
				charging_session["session_added_SoC"] = session_added_SoC
				charging_session["session_cost"] = (
					session_added_SoC * self.evs.b_cap[i] * charging_price +\
					idle_time * self.evs.idle_cost
				)
				charging_session["session_time"] = self.renew_ct
				charging_session["per_step_added_SoC"] = actual_added_SoC / np.ceil(self.renew_ct / self.dt)

				# Debugging output before assertion
				if self.dispatch_results[i]['order'] is not None:
					print(f"Error: EV {i} is charging but has an active order: {self.dispatch_results[i]['order']}")

				assert self.dispatch_results[i]['order'] is None, "To stay charge: serving records should be None"

		# Should process stay_charge_evs first, in case any charger is freed up
		if go_charge_evs:
			self.relocate_to_charge(go_charge_evs)

		# Step 4: Based on actual dispatch results, compute state transition and reward
		rewards = []
		next_states = []
		for i in range(self.N):
			s_i = self.evs.get_state(i)
			reward = self.compute_reward(i, s_i, actions[i])
			next_state = self.state_transition(i, s_i, actions[i])
			self.evs.update_state(i, next_state)

			rewards.append(reward)
			next_states.append(next_state)

		self.agents_trajectory['Reward'][:, self.current_timepoint] = rewards
		self.ep_returns += sum(rewards)
		self.states = self.evs.get_all_states()
  
		# Step 5: Update global states, and metrics
		for i in range(self.N):
			if i in go_charge_evs+stay_charge_evs and self.dispatch_results[i]['cs']:
				self.total_charging_cost += self.dispatch_results[i]['cs']['session_cost']
				self.total_added_soc += self.dispatch_results[i]['cs']["session_added_SoC"]
	
			if i in dispatch_evs and self.dispatch_results[i]['order']:
				self.total_successful_dispatches += 1
				self.total_trip_fare_earned += self.dispatch_results[i]['order']['trip_fare']
	
		self.update_acceptance_rate()
		self.total_trip_requests = len(self.trip_requests.trip_queue)
		
		self.current_timepoint += 1

		done = False
		# if self.current_timepoint >= self.T or all(self.states['SoC'][i] == 0.0 for i in range(self.N)):
		num_requests = int(self.trip_requests.customer_arrivals[self.current_timepoint]) 
		if self.current_timepoint >= self.T:
			done = True
   			# Store final state
			for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
				self.agents_trajectory[s][:, self.current_timepoint] = self.states[s]

		# info = {}
		# if self.current_timepoint >= self.T:
		# 	info = "Episode End."
		# if all(self.states['SoC'][i] == 0.0 for i in range(self.N)):
		# 	info = "All Battery Depleted."

		return np.array(self.states), rewards, done, penalty

	def state_transition(self, ev_i, s_t_i, action_i):
		o_t_i, tau_t_i, theta_t_i = s_t_i
		
		if tau_t_i == 0:  # EV is available for decision making
			if action_i == 0:  # remain-idle action
				theta_t1_i = max(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0) # reduction by time step

				if self.dispatch_results[ev_i]['order']:
					o_t1_i = 1
					session_time = self.dispatch_results[ev_i]['order']['pickup_duration'] + self.dispatch_results[ev_i]['order']['trip_duration']
					tau_t1_i = max(session_time - self.dt, 0)
				else:
					o_t1_i = 0
					tau_t1_i = 0

			elif action_i == 1:  # go-charge or stay charge action
				if self.dispatch_results[ev_i]['cs']:
					o_t1_i = 2
					session_time = self.dispatch_results[ev_i]['cs']['session_time']
					tau_t1_i = max(session_time - self.dt, 0)
					per_step_added_SoC = self.dispatch_results[ev_i]['cs']['per_step_added_SoC']
					theta_t1_i = min(theta_t_i + per_step_added_SoC, 1)
					self.evs.last_charged_time[ev_i] = self.current_timepoint + 1
				else:
					o_t1_i = 0
					tau_t1_i = 0
					theta_t1_i = max(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0)

		elif o_t_i == 1 and tau_t_i >= 1:  # EV is serving an order
			tau_t1_i = max(tau_t_i - self.dt, 0)
			theta_t1_i = max(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0)
			o_t1_i = 1

		elif o_t_i == 2 and tau_t_i >= 1:  # EV is charging
			o_t1_i = 2
			tau_t1_i = max(tau_t_i - self.dt, 0)
			per_step_added_SoC = self.dispatch_results[ev_i]['cs']['per_step_added_SoC']
			theta_t1_i = min(theta_t_i + per_step_added_SoC, 1)
			self.evs.last_charged_time[ev_i] = self.current_timepoint + 1

		# theta_t1_i = 0.8

		return o_t1_i, tau_t1_i, round(theta_t1_i, 4)

	def compute_reward(self, ev_i, s_t_i, action_i):
		o_t_i, tau_t_i, _ = s_t_i
		
		if tau_t_i == 0:
			if action_i == 0:
				if self.dispatch_results[ev_i]['order']:
					r_i = self.dispatch_results[ev_i]['order']['trip_fare']
				else:
					r_i = 0

			elif action_i == 1:
				if self.dispatch_results[ev_i]['cs']:
					r_i = -self.dispatch_results[ev_i]['cs']['session_cost']
				else:
					r_i = 0

		elif o_t_i == 1 and tau_t_i >= 1:
			r_i = 0

		elif o_t_i == 2 and tau_t_i >= 1:
			r_i = 0

		else:
			raise ValueError("Inconsistent state: EV is idle (o_t_i == 0) but has tau_t_i >= 1.")
		
		return r_i

	def render(self):
		print(f"Time step: {self.current_timepoint}, State: {self.states}")
  
	def update_acceptance_rate(self):
		if len(self.trip_requests.trip_queue) == 0:
			self.acceptance_rate = 0
		else:
			self.acceptance_rate = self.total_successful_dispatches / len(self.trip_requests.trip_queue)

	def report_progress(self):
		open_requests = self.trip_requests.update_open_requests(self.current_timepoint*self.dt)
		open_stations = self.charging_stations.update_open_stations()
		total_available_chargers = self.charging_stations.get_dynamic_resource_level()
  
		op_status = self.states["OperationalStatus"]
		idle_evs = [i for i, status in enumerate(op_status) if status == 0]
		serving_evs = [i for i, status in enumerate(op_status) if status == 1]
		charging_evs = [i for i, status in enumerate(op_status) if status == 2]
  
		report = {
			"current_hour": (self.start_hour*60+self.current_timepoint*self.dt)//60,
			"current_minute": (self.start_hour*60+self.current_timepoint*self.dt)%60,
			"open_requests": open_requests,
			"open_charging_stations": open_stations,
			"total_available_chargers_slots": total_available_chargers,
			"EVs_idle": idle_evs,
			"EVs_serving": serving_evs,
			"EVs_charging": charging_evs,
			"total_charging_cost": self.total_charging_cost,
			"total_added_soc": self.total_added_soc,
			"total_successful_dispatches": self.total_successful_dispatches,
			"self.acceptance_rate": self.acceptance_rate,
			"total_trip_fare_earned": self.total_trip_fare_earned,
			"total_penalty": self.total_penalty,
		}
		# print("len(self.trip_requests.trip_queue_list):", len(self.trip_requests.trip_queue))

		print(f"Detailed Status Report at  {(6*60+self.current_timepoint*self.dt)//60}:{(6*60+self.current_timepoint*self.dt)%60}:")
		print(f"  Open Requests: {open_requests}")
		print(f"  Open Charging Stations: {open_stations}")
		print(f"  Total Number of Open Slots: {total_available_chargers}")
		print(f"  Idle EVs: {idle_evs}")
		print(f"  Serving EVs: {serving_evs}")
		print(f"  Charging EVs: {charging_evs}")
		print(f"  Total Charging Cost: {self.total_charging_cost:.2f}")
		print(f"  Total Added SoC: {self.total_added_soc:.2f}")
		print(f"  Total Successful Dispatches: {self.total_successful_dispatches}")
		print(f"  Acceptance Rate: {self.acceptance_rate:.2f}")
		print(f"  Total trip_fare Earned: {self.total_trip_fare_earned:.2f}")
		print(f"  Total Violation Penalty: {self.total_penalty:.2f} ")
  
		return report

	def print_ep_results(self):
		open_count, completed_count = self.trip_requests.count_trip_requests()
		summary = {
			"total_charging_cost": self.total_charging_cost,
			"total_added_soc": self.total_added_soc,
			"total requested trips": open_count + completed_count,
			"total_successful_dispatches": completed_count,
			"acceptance_rate": round(completed_count / (open_count + completed_count),4),
			"total_trip_fare_earned": self.total_trip_fare_earned,
			"total_penalty": self.total_penalty,
			"total_returns": self.ep_returns,
		}	
		return summary

	def random_dispatch(self, dispatch_evs):
	 
		open_requests = self.trip_requests.update_open_requests(self.current_timepoint)
  
		if not open_requests:
			for ev in dispatch_evs:
				self.dispatch_results[ev]['order'] = None
			return

		valid_evs = []
		for ev in dispatch_evs:
			if self.states['SoC'][ev] >= self.evs.min_SoC:
				valid_evs.append(ev)
			else:
				self.dispatch_results[ev]['order'] = None

		if not valid_evs:
			return

		sorted_request_ids = sorted(open_requests, key=lambda req_id: self.trip_requests.trip_queue[req_id]['trip_fare'], reverse=True)

		for req_id in sorted_request_ids:
			request = self.trip_requests.trip_queue[req_id]
			if not valid_evs:
				break

			feasible_evs = [ev for ev in valid_evs if self.states['SoC'][ev] >= request['trip_duration'] * self.evs.energy_consumption[ev] + self.evs.min_SoC]

			if feasible_evs:
				best_ev = self.rng.choice(feasible_evs) 
				valid_evs.remove(best_ev)

				self.dispatch_results[best_ev]['order'] = {
					"trip_fare": request["trip_fare"],
					"pickup_duration": 0,
					"trip_duration": request["trip_duration"],
					"destination": (0,0)
				}
				self.trip_requests.complete_request(req_id)

		for ev in valid_evs:
			self.dispatch_results[ev]['order'] = None

	def relocate_to_charge(self, go_charge_evs):

		open_stations = self.charging_stations.update_open_stations()

		if not open_stations:
			for ev in go_charge_evs:
				self.dispatch_results[ev]['cs'] = None
			return

		if len(go_charge_evs)==0:
			return

		station_id = open_stations[0]
		station_info = self.charging_stations.stations[station_id]
		resource_level = self.charging_stations.get_dynamic_resource_level()

		if resource_level == 0:
			print("Warning: Check the code for open stations—`open_stations` should be empty when `resource_level` is 0.")
			for ev in go_charge_evs:
				self.dispatch_results[ev]['cs'] = None
			return

		self.rng.shuffle(go_charge_evs)
		evs_to_assign = go_charge_evs[:min(resource_level, len(go_charge_evs))]
		# total_assign = len(evs_to_assign)

		for ev in go_charge_evs:
			# print("ev:", ev)
			self.dispatch_results[ev]['cs'] = None  # Default to no charging assignment

			if ev not in evs_to_assign:
				continue  # Skip EVs that were not selected

			SoC_i = self.states["SoC"][ev]
			remaining_cap = max(self.evs.max_SoC - SoC_i, 0)
			session_added_SoC = self.evs.charge_rate[ev] * self.min_ct
			session_added_SoC = min(session_added_SoC, remaining_cap)

			actual_charging_time = session_added_SoC / self.evs.charge_rate[ev]

			# Compute session timing
			# transition_time = self.dt  # Default to 1 time step for travel time; actual travel time can be used if needed
			transition_time = 0
			session_time = transition_time + self.min_ct  # Total session time including transition and charging

			# Idle time is the remaining time after actual charging
			idle_time = session_time - actual_charging_time

			# Energy consumption during transition and idle time
			SoC_drop = self.evs.energy_consumption[ev] * (transition_time + idle_time)
			# actual_added_SoC = max(session_added_SoC - SoC_drop, 0)  # Prevent negative SoC gain
			actual_added_SoC = session_added_SoC - SoC_drop

			if actual_added_SoC <= 0:
				# total_assign = max(total_assign - 1, 0) 
				continue
				# print(SoC_i)
				# print(session_added_SoC)
				# print(SoC_drop)
				# raise ValueError(f"Error: EV {ev} did not add any SoC during the charging session. Please check the parameters or conditions.")

			# Get the real-time charging price
			price_index = int((self.start_hour*60 + self.current_timepoint * self.dt) // 30)
			charging_price = station_info["real_time_prices"][price_index]

			# Update dispatch results for the charging session
			self.dispatch_results[ev]['cs'] = {
				"station_id": station_id,
				"session_added_SoC": session_added_SoC,
				"per_step_added_SoC": actual_added_SoC / np.ceil(session_time / self.dt),
				"session_time": session_time,
				"price": charging_price,
				"session_cost": (
					session_added_SoC * self.evs.b_cap[ev] * charging_price +
					transition_time * self.evs.travel_cost +
					idle_time * self.evs.idle_cost
				)
			}
			self.charging_stations.adjust_occupancy(station_id, -1)
			
		# Update charging station occupancy
		# self.charging_stations.adjust_occupancy(station_id, -total_assign)


	def store_transition(self):
		for agent_id in range(self.N):  # Loop over each agent
			for t in range(self.T):  # Loop over each time step
				# Extract state at time t
				state = np.array([self.agents_trajectory['OperationalStatus'][agent_id, t],
								  self.agents_trajectory['TimeToNextAvailability'][agent_id, t],
								  self.agents_trajectory['SoC'][agent_id, t]])

				# Extract action at time t
				action = self.agents_trajectory['Action'][agent_id, t]

				# Extract reward at time t
				reward = self.agents_trajectory['Reward'][agent_id, t]

				# Extract next state at time t+1 (except for the last step)
				if t < self.T - 1:
					next_state = np.array([self.agents_trajectory['OperationalStatus'][agent_id, t+1],
										   self.agents_trajectory['TimeToNextAvailability'][agent_id, t+1],
										   self.agents_trajectory['SoC'][agent_id, t+1]])
					done = False  # Not done yet
				else:
					next_state = np.zeros_like(state)  # Dummy next state for terminal
					done = True  # Last step of episode

				# Store the experience (appending to global memory)
				self.memory.append((state, action, reward, next_state, done))

	def save_transitions_to_file(self, filename="transitions.pkl"):
		"""Save the agent's transitions to a file."""
		with open(filename, "wb") as f:
			pickle.dump(self.memory, f)
		print(f"Transitions saved to {filename}")
  
		return self.memory

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

	# Define paths
	input_path = "input"
	type_path = f"price{price_type}_demand{demand_type}_SoC{SoC_type}_{total_chargers}for{total_evs}_{start_hour}to24_{resolution}min"
 

	config_filename = os.path.join(os.path.join(input_path, type_path, "train_config.json"))
	
	env = EVChargingEnv(config_filename)  # 3 EVs, total 10 sessions, charging holds 2 sessions
	
	total_episodes = 1
	ep_pay = []
	ep_cost = []
	ep_returns = []
	ep_penalty = []
	for ep in range(total_episodes):
		env.reset()
		env.show_config()

		for _ in range(env.T):
			# Get current state of taxi 0
			o_t_i, tau_t_i, SoC_i = env.evs.get_state(0)
			# Sample an action from the action space
			actions = env.action_space.sample()
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

		ep_pay.append(env.total_trip_fare_earned)
		ep_cost.append(env.total_charging_cost)
		ep_returns.append(env.ep_returns)
		ep_penalty.append(env.total_penalty)
  
	visualize_trajectory(env.agents_trajectory)
 
	serializable_data = {key: value.tolist() for key, value in env.agents_trajectory.items()}
	with open("agents_trajectory.json", "w") as f:
		json.dump(serializable_data, f, indent=4)
	
 
	ep_pay = [round(float(r), 2) for r in ep_pay]
	print("total pay:", ep_pay)
	print("average total pay is:", sum(ep_pay)/total_episodes)	
 
	ep_cost = [round(float(r), 2) for r in ep_cost]
	print("total costs:", ep_cost)
	print("average total costs is:", sum(ep_cost)/total_episodes)	
 
	ep_returns = [round(float(r), 2) for r in ep_returns]
	print("total returns:", ep_returns)
	print("average total returns is:", sum(ep_returns)/total_episodes)	
 
	ep_penalty = [round(float(r), 2) for r in ep_penalty]
	print("total penalty:", ep_penalty)
	print("average total penlaty is:", sum(ep_penalty)/total_episodes)	

	env.close()


if __name__ == "__main__":
	main()
