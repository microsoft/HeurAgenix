import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from src.problems.base.mdp_env import MDPEnv
from src.problems.base.mdp_components import Solution
from src.problems.road_charging.gym_env import RoadCharging


class Env(MDPEnv):
	"""RoadCharging env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
	def __init__(self, data_name: str, **kwargs):
		super().__init__(data_name, RoadCharging, "road_charging")
		self.key_item = "return"
		self.compare = lambda x, y: x - y
		self.construction_steps = self.gym_env.T
		self.online_problem = True
		self.output_dir = "src/problems/road_charging"

	def get_global_data(self):
		return {
			"fleet_size": self.gym_env.N,
			"max_time_steps": self.gym_env.T,
			"time_resolution": self.gym_env.dt,
			"total_chargers": self.gym_env.charging_stations.max_charging_capacity,
			"max_cap": self.gym_env.evs.b_cap,
			"consume_rate": self.gym_env.evs.energy_consumption,
			"charging_rate": self.gym_env.evs.charge_rate,
			# "assign_prob": self.gym_env.rho,
			"customer_arrivals": self.gym_env.trip_requests.customer_arrivals,
			"order_price": self.gym_env.trip_requests.per_minute_rates,
			"charging_price": self.gym_env.charging_stations.get_real_time_prices(),
			"initial_charging_cost": 0
		}
	
	def get_state_data(self):
		return {
			"current_step": self.gym_env.current_timepoint,
			"operational_status":  self.gym_env.states["OperationalStatus"],
			"time_to_next_availability": self.gym_env.states["TimeToNextAvailability"],
			# "ride_lead_time": self.gym_env.obs["RideTime"],
			# "charging_lead_time": self.gym_env.states["ChargingStatus"],
			"battery_soc": self.gym_env.states["SoC"],
			"reward": self.reward,
			"return": self.gym_env.ep_returns,
			"current_solution": self.current_solution
		}

	def validation_solution(self, solution: Solution=None) -> bool:
		return True

	def set_online_mode(self, online_mode: bool):
		self.gym_env.stoch_step = online_mode

	
	def get_observation(self) -> dict:
		return {
			"Sum Battery Soc": sum(self.gym_env.states["SoC"]),
			"Reward": self.reward
		}

	def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
		output_file = open(os.path.join(self.output_dir, "trajectory.json"), "w")
		data_converted = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in self.gym_env.agents_trajectory.items()}  
		json.dump(data_converted, output_file)
		content = super().dump_result(dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
		
		return content

	def show_trajectory(self):
		agents_trajectory = self.gym_env.agents_trajectory
		fname = os.path.join(self.output_dir, "trajectory.png")
  
		N, T = agents_trajectory["Reward"].shape
		chosen_agents = random.sample(range(N), 3)
		time_steps = range(T + 1)

		fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

		for n in chosen_agents:
			plt.subplot(3, 1, 1)
			plt.plot(time_steps, agents_trajectory['SoC'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
			plt.title('Battery State of Charge Over Time')
			plt.ylabel('Battery SoC (%)')
			plt.legend(loc='upper right')

			plt.subplot(3, 1, 2)
			plt.plot(time_steps, agents_trajectory['TimeToNextAvailability'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
			plt.title('Time to Next Availability Over Time')
			plt.ylabel('Time to Availability (Minutes)')
			plt.legend(loc='upper right')

			plt.subplot(3, 1, 3)
			plt.plot(time_steps, agents_trajectory['OperationalStatus'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
			plt.title('Operational Status Over Time')
			plt.xlabel('Time Step')
			plt.ylabel('Status (0=Idle, 1=Serving, 2=Charging)')
			plt.legend(loc='upper right')

		plt.savefig(fname, dpi=300)

  
def main():
	config_filename = "train_config.json"
	
	env = Env(config_filename) 
	env.gym_env.reset(stoch_step=True)

	for _ in range(env.gym_env.T):
			
		actions = env.gym_env.action_space.sample()
		_, _, done, _ = env.gym_env.step(actions)
  
  
	# env.dump_result()
	env.show_trajectory()
  
	print(env.get_global_data())
	print(env.get_state_data())
	env.gym_env.close()

if __name__ == "__main__":
	main()