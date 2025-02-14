import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.cvrp.components import Solution


class Env(BaseEnv):
    """CVRP env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")
        self.node_num, self.distance_matrix, self.depot, self.vehicle_num, self.capacity, self.demands = self.data
        self.construction_steps = self.node_num
        self.key_item = "total_current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(self.state_data["visited_nodes"]) == self.node_num

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path) 
        depot = problem.depots[0] - 1
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
        return node_num, distance_matrix, depot, vehicle_num, capacity, demands

    def init_solution(self) -> Solution:
        return Solution(routes=[[self.depot] for _ in range(self.vehicle_num)], depot=self.depot)

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "node_num" (int): The total number of nodes in the problem.
                - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
                - "vehicle_num" (int): The total number of vehicles.
                - "capacity" (int): The capacity for each vehicle and all vehicles share the same value.
                - "depot" (int): The index for depot node.
                - "demands" (numpy.ndarray): The demand of each node.
        """
        global_data_dict = {
            "node_num": self.node_num,
            "distance_matrix": self.distance_matrix,
            "vehicle_num": self.vehicle_num,
            "capacity": self.capacity,
            "depot": self.depot,
            "demands": self.demands
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): The current set of routes for all vehicles.
                - "visited_nodes" (list[int]): A list of lists representing the nodes visited by each vehicle.
                - "visited_num" (int): Number of nodes visited by each vehicle.
                - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
                - "visited_num" (int): Number of nodes have not been visited by each vehicle.
                - "total_current_cost" (int): The total cost of the current solution.
                - "last_visited" (list[int]): The last visited node for each vehicle.
                - "vehicle_loads" (list[int]): The current load of each vehicle.
                - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        # A list of integers representing the IDs of nodes that have been visited.
        visited_nodes = list(set([node for route in solution.routes for node in route]))

        # A list of integers representing the IDs of nodes that have not yet been visited.
        unvisited_nodes = [node for node in range(self.node_num) if node not in visited_nodes]

        last_visited = []
        vehicle_loads = []
        vehicle_remaining_capacity = []
        total_current_cost = 0
        for vehicle_index in range(self.vehicle_num):
            route = solution.routes[vehicle_index]
            # The cost of the current solution for each vehicle.
            cost_for_vehicle = sum([self.distance_matrix[route[index]][route[index + 1]] for index in range(len(route) - 1)])
            if len(route) > 0:
                cost_for_vehicle += self.distance_matrix[route[-1]][route[0]]
            total_current_cost += cost_for_vehicle
            # The last visited node for each vehicle.
            if len(route) == 0:
                last_visited.append("None")
            else:
                last_visited.append(route[-1])
            # The current load of each vehicle.
            vehicle_loads.append(sum([self.demands[node] for node in route]))
            # The remaining capacity for each vehicle.
            vehicle_remaining_capacity.append(self.capacity - sum([self.demands[node] for node in route]))

        state_dict = {
            "current_solution": solution,
            "visited_nodes": visited_nodes,
            "visited_num": len(visited_nodes),
            "unvisited_nodes": unvisited_nodes,
            "unvisited_num": len(unvisited_nodes),
            "total_current_cost": total_current_cost,
            "last_visited": last_visited,
            "vehicle_loads": vehicle_loads,
            "vehicle_remaining_capacity": vehicle_remaining_capacity,
            "validation_solution": self.validation_solution
        }
        return state_dict

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
                if not (0 <= node < self.node_num):
                    return False

        # Check uniqueness
        all_nodes = [node for route in solution.routes for node in route if node != self.depot] + [self.depot]
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route in solution.routes:
            # Check include depot
            if self.depot not in route:
                return False

            # Check vehicle load capacity constraints
            load = sum(self.demands[node] for node in route)
            if load > self.capacity:
                return False

        return True

    def get_observation(self) -> dict:
        return {
            "Visited Node Num": self.state_data["visited_num"],
            "Current Cost": self.state_data["total_current_cost"],
            "Fulfilled Demands": sum([self.demands[node] for node in self.state_data["visited_nodes"]])
        }

    def dump_result(self, dump_trajectory: bool=True, result_file: str="result.txt") -> str:
        content_dict = {
            "node_num": self.node_num,
            "visited_num": self.state_data["visited_num"]
        }
        content = super().dump_result(content_dict, dump_trajectory, result_file)
        return content
