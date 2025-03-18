import numpy as np

class EVFleet:
	def __init__(self, config=None):
		if config is None:
			config = {
				"N": 5,
				"SoC_drop_per_min": [0.002] * 5,
				"SoC_added_per_min": [0.0167] * 5,
				"battery_capacity_kwh": [62] * 5,
				"init_SoCs": [0.8] * 5,
				"min_SoC": 0.1,
				"max_SoC": 1.0,
				"travel_cost": 3, # per minute
				"idle_cost": 3 # per minute
			}

		self.N = config.get("N", 5)
		self.energy_consumption = config.get("SoC_drop_per_min", [0.002] * self.N)
		self.charge_rate = config.get("SoC_added_per_min", [0.013] * self.N)
		self.b_cap = config.get("battery_capacity_kwh", [62] * self.N)
		self.init_SoCs = np.array(config.get("init_SoCs", [0.8] * self.N))
		self.min_SoC = config.get("min_SoC", 0.1)
		self.max_SoC = config.get("max_SoC", 0.8)
		# self.travel_cost = config.get("travel_cost", 0.8)
		# self.idle_cost = config.get("idle_cost", 0.8)
		self.travel_cost = 0.0
		self.idle_cost = 0.0

		self.all_states = None  # Initialize placeholder
		

	def reset(self, rng):
		self.rng = rng
		
		self.all_states = {
			"OperationalStatus": np.zeros(self.N, dtype=int),  # 0: idle, 1: serving, 2: charging
			"TimeToNextAvailability": np.zeros(self.N, dtype=int),
			"SoC": self.init_SoCs.copy(),  # Ensure a fresh copy
		}

		self.last_charged_time = {i:-1 for i in range(self.N)}

		return self.all_states

	def reset_init_SoCs(self):
		self.init_SoCs = np.round(self.rng.uniform(0.75, 0.8, size=self.N), decimals=4)
		self.all_states["SoC"] = self.init_SoCs.copy()
  

	def update_state(self, ev_index, new_state):
	
		self.all_states["OperationalStatus"][ev_index] = new_state[0]
		self.all_states["TimeToNextAvailability"][ev_index] = new_state[1]
		self.all_states["SoC"][ev_index] = new_state[2]

	def get_state(self, ev_index):
		
		return [
				self.all_states[key][ev_index] 
				for key in ["OperationalStatus", "TimeToNextAvailability", "SoC"]
			]
	
	def get_all_states(self):
		
		return self.all_states

	def get_last_charged_time(self, ev):
		return self.last_charged_time[ev]


