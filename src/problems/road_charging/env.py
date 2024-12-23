import os
import json
import numpy as np
from src.problems.base.mdp_env import MDPEnv
from src.problems.base.mdp_components import Solution
from src.problems.road_charging.gym_env import RoadCharging


class Env(MDPEnv):
    """RoadCharging env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    # def __init__(self, data_name: str, constrain: bool = False, **kwargs):
    #     if constrain:
    #         super().__init__(data_name, ConstrainAction, "constrained_road_charging")
    #     else:
    #         super().__init__(data_name, RoadCharging, "road_charging")
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, RoadCharging, "road_charging")
        self.key_item = "return"
        self.compare = lambda x, y: y - x
        self.construction_steps = self.gym_env.k

    def get_global_data(self):
        return {
            "fleet_size": self.gym_env.n,
            "max_time_steps": self.gym_env.k,
            "total_chargers": self.gym_env.m,
            "max_cap": self.gym_env.max_cap,
            "consume_rate": self.gym_env.d_rates,
            "charging_rate": self.gym_env.c_rates,
            "assign_prob": self.gym_env.rho,
            "order_price": self.gym_env.w,
            "charging_price": self.gym_env.p,
            "initial_charging_cost": self.gym_env.h
        }
    
    def get_state_data(self):
        return {
            "current_step": self.gym_env.obs["TimeStep"][0],
            "ride_lead_time": self.gym_env.obs["RideTime"],
            "charging_lead_time": self.gym_env.obs["ChargingStatus"],
            "battery_soc": self.gym_env.obs["SoC"],
            "reward": self.reward,
            "return": self.gym_env.ep_return,
            "current_solution": self.current_solution
        }

    def validation_solution(self, solution: Solution=None) -> bool:
        return True

    def dump_result(self, dump_trajectory = True):
        output_file = open(os.path.join(self.output_dir, "trajectory.json"), "w")
        data_converted = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in self.gym_env.trajectories.items()}  
        json.dump(data_converted, output_file)
        return super().dump_result(dump_trajectory)
