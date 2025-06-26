import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.cvrp.components import Solution


class Env(BaseEnv):
    """CVRP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "total_current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set([node for route in self.current_solution.routes for node in route])) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path)
        depot = problem.depots[0] - 1
        if problem.edge_weight_type == "EUC_2D":
            node_coords = problem.node_coords
            node_num = len(node_coords)
            distance_matrix = np.zeros((node_num, node_num))
            for i in range(node_num):
                for j in range(node_num):
                    if i != j:
                        x1, y1 = node_coords[i + 1]
                        x2, y2 = node_coords[j + 1]
                        distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:
            distance_matrix = nx.to_numpy_array(problem.get_graph())
            node_num = len(distance_matrix)
        if os.path.basename(data_path).split(".")[0].split("-")[-1][0] == "k":
            vehicle_num = int(os.path.basename(data_path).split(".")[0].split("-")[-1][1:])
        elif open(data_path).readlines()[-1].strip().split(" : ")[0] == "VEHICLE":
            vehicle_num = int(open(data_path).readlines()[-1].strip().split(" : ")[-1])
        else:
            raise NotImplementedError("Vehicle number error")
        capacity = problem.capacity
        demands = np.array(list(problem.demands.values()))
        return {"node_num": node_num, "distance_matrix": distance_matrix, "depot": depot, "vehicle_num": vehicle_num, "capacity": capacity, "demands": demands}

    def init_solution(self) -> Solution:
        return Solution(routes=[[self.instance_data["depot"]] for _ in range(self.instance_data["vehicle_num"])], depot=self.instance_data["depot"])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        total_current_cost = 0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            # The cost of the current solution for each vehicle.
            cost_for_vehicle = sum([self.instance_data["distance_matrix"][route[index]][route[index + 1]] for index in range(len(route) - 1)])
            if len(route) > 0:
                cost_for_vehicle += self.instance_data["distance_matrix"][route[-1]][route[0]]
            total_current_cost += cost_for_vehicle
        return total_current_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node existence: Each node in each route must be within the valid range.
            2. Uniqueness: Each node (except for the depot) must only be visited once across all routes.
            3. Include depot: Each route must include at the depot.
            4. Capacity constraints: The load of each vehicle must not exceed its capacity.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.routes, list):
            return False

        # Check node existence
        for route in solution.routes:
            for node in route:
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

        # Check uniqueness
        all_nodes = [node for route in solution.routes for node in route if node != self.instance_data["depot"]] + [self.instance_data["depot"]]
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route in solution.routes:
            # Check include depot
            if self.instance_data["depot"] not in route:
                return False

            # Check vehicle load capacity constraints
            load = sum(self.instance_data["demands"][node] for node in route)
            if load > self.instance_data["capacity"]:
                return False

        return True
