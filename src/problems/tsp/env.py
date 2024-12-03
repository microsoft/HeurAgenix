import numpy as np
import networkx as nx
import tsplib95
from src.problems.base.env import BaseEnv
from src.problems.tsp.components import Solution


class Env(BaseEnv):
    """TSP env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "tsp")
        self.node_num, self.distance_matrix = self.data
        self.construction_steps = self.node_num
        self.key_item = "current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set(self.current_solution.tour)) == self.node_num

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        node_num = len(distance_matrix)
        return node_num, distance_matrix

    def init_solution(self) -> None:
        return Solution(tour=[])

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "node_num" (int): The total number of nodes in the problem.
                - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        """

        global_data_dict = {
            "distance_matrix": self.distance_matrix,
            "node_num": self.node_num
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "visited_nodes" (list[int]): A list of integers representing the IDs of nodes that have been visited.
                - "visited_num" (int): Number of nodes that have been visited.
                - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
                - "unvisited_num" (int): Number of nodes that have not been visited.
                - "current_cost" (int): The total cost of current solution. The cost to return to the starting point is not included until the path is fully constructed.
                - "last_visited" (int): The last visited node.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        # A list of integers representing the IDs of nodes that have been visited.
        visited_nodes = solution.tour

        # A list of integers representing the IDs of nodes that have not yet been visited.
        unvisited_nodes = [node for node in range(self.node_num) if node not in visited_nodes]

        # The total cost of current solution.
        current_cost = sum([self.distance_matrix[solution.tour[index]][solution.tour[index + 1]] for index in range(len(solution.tour) - 1)])
        if len(solution.tour) > 0:
            current_cost += self.distance_matrix[solution.tour[-1]][solution.tour[0]]

        # The last visited node 
        last_visited = None if not solution.tour else solution.tour[-1]

        state_data_dict = {
            "current_solution": solution,
            "visited_nodes": visited_nodes,
            "visited_num": len(visited_nodes),
            "unvisited_nodes": unvisited_nodes,
            "unvisited_num": len(unvisited_nodes),
            "current_cost": current_cost,
            "last_visited": last_visited,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node Existence: Each node in the solution must exist within the problem instance's range of nodes.
            2. Uniqueness: No node is repeated within the solution path, ensuring that each node is visited at most once.
            3. Connectivity: Each edge (from one node to the next) must be connected, i.e., not marked as infinite distance in the distance matrix.
        """
        node_set = set()
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.tour, list):
            return False
        if solution is not None and solution.tour is not None:
            for index, node in enumerate(solution.tour):
                # Check node existence
                if not (0 <= node < self.node_num):
                    return False

                # Check uniqueness
                if node in node_set:
                    return False
                node_set.add(node)

                # Check connectivity if not the last node
                if index < len(solution.tour) - 1:
                    next_node = solution.tour[index + 1]
                    if self.distance_matrix[node][next_node] == np.inf:
                        return False
        return True

    def get_observation(self) -> dict:  
        return {
            "Visited Node": self.state_data["visited_num"],
            "Current Cost": self.state_data["current_cost"]
        }

    def dump_result(self, dump_trajectory: bool=True) -> str:
        content_dict = {
            "node_num": self.node_num,
            "visited_num": self.state_data["visited_num"]
        }
        content = super().dump_result(content_dict, dump_trajectory)
        return content
