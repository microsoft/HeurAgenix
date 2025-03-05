import os
import numpy as np
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.vrptw.components import Solution


class Env(BaseEnv):
    """VRPTW env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "vrptw")
        self.node_num, self.distance_matrix, self.depot, self.vehicle_num, self.capacity, self.time_windows, self.service_times, self.demands = self.data
        self.construction_steps = self.node_num
        self.key_item = "total_distance_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(self.state_data["visited_nodes"]) == self.node_num

    def load_data(self, data_path: str) -> None:
        lines = open(data_path).readlines()
        depot = 0

        vehicle_num = int(lines[4].strip().split(" ")[0])
        capacity = int(lines[4].strip().split(" ")[-1])
        coords = []
        demands = []
        time_windows = []
        service_times = []
        for line in lines[9:]:
            nums = [int(element) for element in line.strip().split(" ") if element != '']
            _, x, y, demand, ready_time, due_time, service_time = nums
            coords.append([x, y])
            demands.append(demand)
            time_windows.append([ready_time, due_time])
            service_times.append(service_time)
        node_num = len(coords)
        distance_matrix = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(node_num):
                distance_matrix[i, j] = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
        return node_num, distance_matrix, depot, vehicle_num, capacity, np.array(time_windows), np.array(service_times), np.array(demands)

    def init_solution(self) -> Solution:
        return Solution(routes=[[self.depot, self.depot] for _ in range(self.vehicle_num)], depot=self.depot)

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "node_num" (int): The total number of nodes in the problem.
                - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
                - "vehicle_num" (int): The total number of vehicles.
                - "capacity" (int): The capacity for each vehicle and all vehicles share the same value.
                - "depot" (int): The index for depot node.
                - "time_windows" (N * 2 numpy.ndarray): The ready and due time for each node.
                - "service_times" (numpy.ndarray): The service time cost of each node.
                - "demands" (numpy.ndarray): The demand of each node.
        """
        global_data_dict = {
            "node_num": self.node_num,
            "distance_matrix": self.distance_matrix,
            "vehicle_num": self.vehicle_num,
            "capacity": self.capacity,
            "depot": self.depot,
            "time_windows": self.time_windows,
            "service_times": self.service_times,
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
                - "distance_for_vehicle(list[int])": The distance cost for each vehicle.
                - "total_distance_cost" (int): The total cost of the current solution.
                - "detailed_time" (list[list[list]]): A detailed list to record all time-related information for each vehicle and its tasks:  
                    - The outermost list corresponds to each vehicle, where `len(detailed_time)` equals the total number of vehicles.  
                    - The second-level list corresponds to the sequence of tasks (or nodes) visited by a vehicle. For example, `len(detailed_time[i])` represents the number of tasks completed by vehicle `i`.  
                    - The innermost list contains 6 elements that describe the detailed time information for a specific task:  
                        1. **Task ID** (int): The unique identifier of the task or node.  
                        2. **Arrival Time** (float): The time when the vehicle arrives at the task location.  
                        3. **Start Time** (float): The time when the vehicle starts servicing the task.  
                        4. **Finish Time** (float): The time when the vehicle finishes servicing the task.  
                        5. **Ready Time** (float): The earliest allowable start time for servicing the task (from the problem definition).  
                        6. **Due Time** (float): The latest allowable time by which the task must be serviced (from the problem definition).  
                - "last_visited" (list[int]): The last visited node for each vehicle.
                - "vehicle_loads" (list[int]): The current load of each vehicle.
                - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
                - "validation_single_route" (callable): def validation_single_route(route: list[int]) -> bool: function to check whether the single route is valid, including capacity check and time window check. If this route is valid, will return detailed_time for this route only(list of list), otherwise will return False.
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
        distance_for_vehicle = []
        routes_with_detailed_time = []
        for route in solution.routes:
            detailed_time = self.validation_single_route(route)
            routes_with_detailed_time.append(detailed_time)
            last_visited.append(route[-1])
            distance_for_vehicle.append(sum([self.distance_matrix[route[index]][route[index + 1]] for index in range(len(route) - 1)]))

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
            "distance_for_vehicle": distance_for_vehicle,
            "total_distance_cost": sum(distance_for_vehicle),
            "detailed_time": routes_with_detailed_time,
            "last_visited": last_visited,
            "vehicle_loads": vehicle_loads,
            "vehicle_remaining_capacity": vehicle_remaining_capacity,
            "validation_single_route": self.validation_single_route,
            "validation_solution": self.validation_solution
        }
        return state_dict

    def validation_single_route(self, route: list[int]) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node existence: Each node in each route must be within the valid range.
            2. Timely: All nodes must be serviced during ready time and due time.
            3. Include depot: Each route must include at the depot.
            4. Capacity constraints: The load of each vehicle must not exceed its capacity.
        """
        load = 0
        arrive_time = 0
        detailed_time = []
        # Check include depot
        if route[0] != self.depot or route[-1] != self.depot:
            return False
        for node_index, node in enumerate(route):
            # Check node existence
            if not (0 <= node < self.node_num):
                return False
            if node_index != 0:
                distance = self.distance_matrix[route[node_index - 1], node]
                arrive_time = finish_time + distance
            start_time = max(arrive_time, self.time_windows[node][0])
            # Check timely
            if start_time >= self.time_windows[node][1]:
                return False
            finish_time = start_time + self.service_times[node]
            detailed_time.append([node, arrive_time, start_time, finish_time, self.time_windows[node][0], self.time_windows[node][1]])
            load += self.demands[node]
            # Check vehicle load capacity constraints
            if load > self.capacity:
                return False
        return detailed_time
        


    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node existence: Each node in each route must be within the valid range.
            2. Uniqueness: Each node (except for the depot) must only be visited once across all routes.
            3. Timely: All nodes must be serviced during ready time and due time.
            4. Include depot: Each route must include at the depot.
            5. Capacity constraints: The load of each vehicle must not exceed its capacity.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.routes, list):
            return False

        # Check uniqueness
        all_nodes = [node for route in solution.routes for node in route if node != self.depot] + [self.depot]
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route_index, route in enumerate(solution.routes):
            # Check include depot
            # Check node existence
            # Check timely
            # Check vehicle load capacity constraints
            if not self.validation_single_route(route):
                return False

        return True

    def get_observation(self) -> dict:
        return {
            "Visited Node Num": self.state_data["visited_num"],
            "Current Cost": self.state_data["total_current_cost"],
            "Fulfilled Demands": sum([self.demands[node] for node in self.state_data["visited_nodes"]])
        }

    def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content_dict = {
            "node_num": self.node_num,
            "visited_num": self.state_data["visited_num"]
        }
        content = super().dump_result(content_dict=content_dict, dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
        return content
