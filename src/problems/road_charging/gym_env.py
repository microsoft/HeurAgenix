import gym
from gym import spaces
import numpy as np
import json
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
from src.problems.road_charging.utils import load_file, visualize_trajectory
from src.problems.road_charging.ChargingStations import ChargingStations
from src.problems.road_charging.TripRequests import TripRequests
from src.problems.road_charging.EVFleet import EVFleet

class RoadCharging(gym.Env):
	def __init__(self, config_fname: str):
	
		config = load_file(config_fname)

		self.T = config.get("total_time_steps")
		self.dt = config.get("time_step_minutes")
		self.N = config.get("total_evs")
		self.min_ct = config.get("committed_charging_block_minutes")
		self.renew_ct = config.get("renewed_charging_block_minutes")
		self.start_hour = config.get("operation_start_hour")
		self.demand_scaling = config.get("demand_scaling")
		
		self.evs = EVFleet(config["ev_params"])
		self.trip_requests = TripRequests(config["trip_params"])
		self.charging_stations = ChargingStations(config["charging_params"])
		
		self.other_env_params = config.get("env_params", {})

		self.memory = deque(maxlen=10000)  # Store all transitions here

		# Action space: 2 actions (remain-idle, go-charge)
		self.action_space = spaces.MultiBinary(self.N)

		# State space: For each EV, (operational-status, time-to-next-availability, SoC, location)
		self.observation_space = spaces.Dict({
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
  
		self.trip_requests.rescale_customer_arrivals(int(self.N*self.demand_scaling))
		# self.trip_requests.customer_arrivals = [int(np.ceil(x/200*self.N)) for x in self.trip_requests.customer_arrivals]
		
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
		self.ep_returns = 0  # Accumulate total returns over T timesteps
		self.charging_cost = 0.0
		self.added_soc = 0.0
		self.fare_earned = 0.0
		self.trip_requests_count = 0
		self.successful_dispatches = 0
		self.complete_rate = 0.0
		self.step_complete_rate = [0.0]*self.T
		self.step_trip_time = [0.0]*self.T
		self.step_requested_trips = [0.0]*self.T
		self.step_successful_dispatches = [0.0]*self.T
		self.driver_earnings = [0.0]*self.N
		self.driver_trip_time = [0.0]*self.N
		self.driver_idle_time = [0.0]*self.N		
		

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

		error_message = ""

		for i in range(self.N):
			# o_t_i, tau_t_i, SoC_i = self.evs.get_state(i)  # Get EV state
			tau_t_i = self.evs.get_state(i)[1] 

			if actions[i] == 1:  # Charging action
				if tau_t_i >= 1:
					
					error_message += f"Step {self.current_timepoint}: EV{i} is requesting charging while busy.\n"
				
		max_charging_cap = self.charging_stations.max_charging_capacity
		if sum(actions) > max_charging_cap:
			error_message += f"Step {self.current_timepoint}: Charging demand {sum(actions)} exceeds charging capacity {max_charging_cap}.\n"
		

		return error_message


	def step(self, actions):
		
		error_message = self.check_action_feasibility(actions)
		if len(error_message) > 0:
			# raise Exception(error_message)
			print(error_message)


  		# Step 1: Take in new customer requests
		if self.stoch_step:
			num_requests = int(self.trip_requests.customer_arrivals[int(self.start_hour*(60/self.dt))+self.current_timepoint]) 
			self.trip_requests.sample_requests(num_requests, self.current_timepoint)
		else:
			num_requests = sum(1 for req in self.trip_requests.trip_queue.values() if req["raised_time"] == self.current_timepoint)
		self.step_requested_trips[self.current_timepoint] = num_requests

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
					raise Exception(f"Warning: Step {self.current_timepoint}: EV {i} has no active charging session but is requesting to renew charging.\n")
					# print(f"Warning: EV {i} has no active charging session.")
					# error_message += f"Warning: Step {self.current_timepoint}: EV {i} has no active charging session but is requesting to renew charging.\n"
					continue  # Skip this EV

				charging_session = self.dispatch_results[i]['cs']
				
				actual_charging_time = session_added_SoC / self.evs.charge_rate[i]
				idle_time = self.renew_ct - actual_charging_time
				idle_time = max(idle_time, 0)  # Ensure no negative idle time
				self.driver_idle_time[i] += idle_time

				# Renew charging session
				charging_price = charging_session.get("price")
				charging_session["session_added_SoC"] = session_added_SoC
				charging_session["session_cost"] = (
					session_added_SoC * self.evs.b_cap[i] * charging_price +\
					idle_time * self.evs.idle_cost
				)
				charging_session["session_time"] = self.renew_ct
				charging_session["per_step_added_SoC"] = session_added_SoC / np.ceil(self.renew_ct / self.dt)

				# Debugging output before assertion
				if self.dispatch_results[i]['order'] is not None:
					raise Exception(f"Error: Step {self.current_timepoint}: EV {i} is charging but has an active order: {self.dispatch_results[i]['order']}")
					# print(f"Error: EV {i} is charging but has an active order: {self.dispatch_results[i]['order']}")
					# error_message += f"Error: Step {self.current_timepoint}: EV {i} is charging but has an active order: {self.dispatch_results[i]['order']}"
				# assert self.dispatch_results[i]['order'] is None, "To stay charge: serving records should be None"

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
				self.charging_cost += self.dispatch_results[i]['cs']['session_cost']
				self.added_soc += self.dispatch_results[i]['cs']["session_added_SoC"]
	
			if i in dispatch_evs and self.dispatch_results[i]['order']:
				self.successful_dispatches += 1
				self.fare_earned += self.dispatch_results[i]['order']['trip_fare']
				self.driver_earnings[i] += self.dispatch_results[i]['order']['trip_fare']
				self.driver_trip_time[i] += self.dispatch_results[i]['order']['trip_duration']
				self.step_trip_time[self.current_timepoint] += self.dispatch_results[i]['order']['trip_duration']
				self.step_successful_dispatches[self.current_timepoint] += 1
				self.step_complete_rate[self.current_timepoint] = (
							round(self.step_successful_dispatches[self.current_timepoint] / self.step_requested_trips[self.current_timepoint],4)
							if self.step_requested_trips[self.current_timepoint] != 0 else 0
						)

		self.complete_rate = round(self.successful_dispatches / len(self.trip_requests.trip_queue),4) if len(self.trip_requests.trip_queue) != 0 else 0
		self.trip_requests_count = len(self.trip_requests.trip_queue)
		
		self.current_timepoint += 1

		done = False
		# if self.current_timepoint >= self.T or all(self.states['SoC'][i] == 0.0 for i in range(self.N)):
		if self.current_timepoint >= self.T:
			done = True
   			# Store final state
			for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
				self.agents_trajectory[s][:, self.current_timepoint] = self.states[s]

		return np.array(self.states), rewards, done, error_message

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
			"total_charging_cost": self.charging_cost,
			"total_added_soc": self.added_soc,
			"total_successful_dispatches": self.successful_dispatches,
			"self.complete_rate": self.complete_rate,
			"total_trip_fare_earned": self.fare_earned,
		}

		print(f"Detailed Status Report at  {(6*60+self.current_timepoint*self.dt)//60}:{(6*60+self.current_timepoint*self.dt)%60}:")
		print(f"  Open Requests: {open_requests}")
		print(f"  Open Charging Stations: {open_stations}")
		print(f"  Total Number of Open Slots: {total_available_chargers}")
		print(f"  Idle EVs: {idle_evs}")
		print(f"  Serving EVs: {serving_evs}")
		print(f"  Charging EVs: {charging_evs}")
		print(f"  Total Charging Cost: {self.charging_cost:.2f}")
		print(f"  Total Added SoC: {self.added_soc:.2f}")
		print(f"  Total Successful Dispatches: {self.successful_dispatches}")
		print(f"  Acceptance Rate: {self.complete_rate:.2f}")
		print(f"  Total trip_fare Earned: {self.fare_earned:.2f}")
  
		return report

	def print_ep_results(self):
		open_count, completed_count = self.trip_requests.count_trip_requests()
		summary = {
				"total_returns": self.ep_returns,
				"total_fare_earned": self.fare_earned,
				"total_charging_cost": self.charging_cost,
				"total_added_soc": self.added_soc,
				"completion_rate": round(completed_count / (open_count + completed_count), 4),
				"total_requested_trips": open_count + completed_count,
				"successful_dispatches": completed_count,
				"total_idle_time": sum(self.driver_idle_time),
				"driver_earnings": list(self.driver_earnings),
				"driver_idle_time": list(self.driver_idle_time),
				"driver_trip_time": list(self.driver_trip_time),
				"step_completion_rate": list(self.step_complete_rate),
				"step_trip_time": list(self.step_trip_time)
			}

		return summary

	def random_dispatch(self, dispatch_evs):
	 
		open_requests = self.trip_requests.update_open_requests(self.current_timepoint)
  
		if not open_requests:
			for ev in dispatch_evs:
				self.dispatch_results[ev]['order'] = None
				self.driver_idle_time[ev]+=self.dt
			return

		valid_evs = []
		for ev in dispatch_evs:
			if self.states['SoC'][ev] >= self.evs.min_SoC:
				valid_evs.append(ev)
			else:
				self.dispatch_results[ev]['order'] = None
				self.driver_idle_time[ev]+=self.dt

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
			self.driver_idle_time[ev]+=self.dt

	def relocate_to_charge(self, go_charge_evs):

		open_stations = self.charging_stations.update_open_stations()

		if not open_stations:
			for ev in go_charge_evs:
				self.dispatch_results[ev]['cs'] = None
				self.driver_idle_time[ev]+=self.dt
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
				self.driver_idle_time[ev]+=self.dt
			return

		self.rng.shuffle(go_charge_evs)
		evs_to_assign = go_charge_evs[:min(resource_level, len(go_charge_evs))]
		# total_assign = len(evs_to_assign)

		for ev in go_charge_evs:
			# print("ev:", ev)
			self.dispatch_results[ev]['cs'] = None  # Default to no charging assignment

			if ev not in evs_to_assign:
				self.driver_idle_time[ev]+=self.dt
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
			self.driver_idle_time[ev] += idle_time

			# Get the real-time charging price
			price_index = int((self.start_hour*60 + self.current_timepoint * self.dt) // 30)
			charging_price = station_info["real_time_prices"][price_index]

			# Update dispatch results for the charging session
			self.dispatch_results[ev]['cs'] = {
				"station_id": station_id,
				"session_added_SoC": session_added_SoC,
				"per_step_added_SoC": session_added_SoC / np.ceil(session_time / self.dt),
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
	start_hour = 0

	price_type = 1
	demand_type = 1
	SoC_type = 1
 

	# Define paths
	input_path = "input"
	type_path = f"price{price_type}_demand{demand_type}_SoC{SoC_type}_{total_chargers}for{total_evs}_{start_hour}to24_{resolution}min"
 
	config_filename = os.path.join(os.path.join(input_path, type_path, "train_config.json"))
	
	env = RoadCharging(config_filename)  # 3 EVs, total 10 sessions, charging holds 2 sessions
	
	num_episodes = 20
	ep_returns_list = []
	charging_cost_list = []
	added_soc_list = []
	fare_earned_list = []
	complete_rate_list = []

	driver_earnings_list = []
	driver_trip_time_list = []
	driver_idle_time_list = []

	step_complete_rates = []
	step_trip_times = []
 
	for ep in range(num_episodes):
		env.reset(stoch_step=True)
		# env.show_config()

		for _ in range(env.T):
			# Get current state of taxi 0
			# o_t_i, tau_t_i, SoC_i = env.evs.get_state(0)
			# Sample an action from the action space
			actions = env.action_space.sample()
			action = actions[0]
	
			o_t_i, tau_t_i, SoC_i = env.evs.get_state(0)
			# Print the current timepoint and state information
			print(f"--- Timepoint {env.current_timepoint} ---")
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
			print("Lead Time:", tau)
			print("SoCs:", SoCs)
			print("Actions:", actions)
		
			# Take a simulation step
			_, _, done, _ = env.step(actions)
   
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
		ep_returns_list.append(env.ep_returns)
		charging_cost_list.append(env.charging_cost)
		added_soc_list.append(env.added_soc)
		fare_earned_list.append(env.fare_earned)
		complete_rate_list.append(env.complete_rate)

		driver_earnings_list.append(env.driver_earnings)
		driver_trip_time_list.append(env.driver_trip_time)
		driver_idle_time_list.append(env.driver_idle_time)

		step_complete_rates.append(env.step_complete_rate)
		step_trip_times.append(env.step_trip_time)

	# Compute average statistics
	avg_ep_returns = np.mean(ep_returns_list)
	avg_charging_cost = np.mean(charging_cost_list)
	avg_added_soc = np.mean(added_soc_list)
	avg_fare_earned = np.mean(fare_earned_list)
	avg_complete_rate = np.mean(complete_rate_list)

	avg_driver_earnings = np.mean(driver_earnings_list, axis=0)
	avg_driver_trip_time = np.mean(driver_trip_time_list, axis=0)
	avg_driver_idle_time = np.mean(driver_idle_time_list, axis=0)

	avg_step_complete_rate = np.mean(step_complete_rates, axis=0)
	avg_step_trip_time = np.mean(step_trip_times, axis=0)
 
	# Print averaged results
	print(f"Avg Ep Returns: {avg_ep_returns}, Avg Charging Cost: {avg_charging_cost}, Avg Added SoC: {avg_added_soc},")
	print(f"Avg Fare Earned: {avg_fare_earned}, Avg Complete Rate: {avg_complete_rate}")

	print(f"Avg Driver Earnings: {avg_driver_earnings}, Sum: {sum(avg_driver_earnings)}")
	print(f"Avg Driver Trip Time: {avg_driver_trip_time}, Sum: {sum(avg_driver_trip_time)}")
	print(f"Avg Driver Idle Time: {avg_driver_idle_time}, Sum: {sum(avg_driver_idle_time)}")

	# Plot averaged step metrics
	fig, ax1 = plt.subplots()

	# ax1.plot(avg_step_complete_rate, label="Avg Step Complete Rate", color="tab:blue")
	# ax1.step(range(len(avg_step_complete_rate)), avg_step_complete_rate, 
	#      label="Avg Step Complete Rate", color="tab:orange", alpha=0.5, linewidth=1, where="post")
	ax1.plot(avg_step_complete_rate, label="Avg Step Complete Rate", color="tab:orange", alpha=0.5, linewidth=1, )
	ax1.set_xlabel("Time Steps")
	ax1.set_ylabel("Complete Rate", color="tab:orange")
	ax1.tick_params(axis="y", labelcolor="tab:orange")

	ax2 = ax1.twinx()
	# ax2.plot(avg_step_trip_time, label="Avg Step Trip Time", color="tab:orange")
	ax2.step(range(len(avg_step_trip_time)), avg_step_trip_time, linestyle="dashed", linewidth=1, label="Avg Step Trip Time", color="tab:blue", where="post")
	ax2.set_ylabel("Trip Time", color="tab:blue")
	ax2.tick_params(axis="y", labelcolor="tab:blue")

	plt.title(f"Averaged Step Metrics Over {num_episodes} Episodes")
	fig.tight_layout()
	plt.show()

	
	visualize_trajectory(env.agents_trajectory)
 
	serializable_data = {key: value.tolist() for key, value in env.agents_trajectory.items()}
	with open("agents_trajectory.json", "w") as f:
		json.dump(serializable_data, f, indent=4)
	
 

	env.close()


if __name__ == "__main__":
	main()
