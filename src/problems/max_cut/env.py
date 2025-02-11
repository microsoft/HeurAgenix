import os
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.max_cut.components import Solution


class Env(BaseEnv):
    """MaxCut env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "max_cut")
        self.node_num, self.weight_matrix = self.data
        self.construction_steps = self.node_num
        self.key_item = "current_cut_value"
        self.compare = lambda x, y: x - y

    @property
    def is_complete_solution(self) -> bool:
        return len(self.current_solution.set_a) + len(self.current_solution.set_b) == self.node_num

    def load_data(self, data_path: str) -> tuple:
        with open(data_path) as file:
            node_num = int(file.readline().split(" ", 1)[0])
            weight_matrix = np.zeros((node_num, node_num))
            for row in file:
                node_1, node_2, weight = [int(e) for e in row.strip("\n").split()]
                weight_matrix[node_1 - 1][node_2 - 1] = weight
                weight_matrix[node_2 - 1][node_1 - 1] = weight
        return node_num, weight_matrix

    def init_solution(self) -> Solution:
        return Solution(set_a=set(), set_b=set())

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "node_num" (int): The total number of vertices in the graph.
                - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        """

        global_data_dict = {
            "node_num": self.node_num,
            "weight_matrix": self.weight_matrix,
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "set_a_count" (int): The number of nodes in set A of the current partition.
                - "set_b_count" (int): The number of nodes in set B of the current partition.
                - "selected_nodes" (set[int]): The set of selected nodes.
                - "selected_num" (int): The number of nodes have been selected.
                - "unselected_nodes" (set[int]): The set of unselected nodes.
                - "unselected_num" (int): The number of nodes have not been selected.
                - "current_cut_value" (int or float): The total weight of edges between set A and set B in the current solution.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution
        
        # Calculate the current cut value
        current_cut_value = 0
        for node_a in solution.set_a:
            for node_b in solution.set_b:
                current_cut_value += self.weight_matrix[node_a][node_b]

        state_data_dict = {
            "current_solution": solution,
            "set_a_count": len(solution.set_a),
            "set_b_count": len(solution.set_b),
            "selected_nodes": solution.set_a.union(solution.set_b),
            "selected_num": len(solution.set_a) + len(solution.set_b),
            "unselected_nodes": set(range(self.node_num)) - solution.set_a - solution.set_b,
            "unselected_num": self.node_num - len(solution.set_a) - len(solution.set_b),
            "current_cut_value": current_cut_value,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Non-repeat: No nodes in both set A and set B
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.set_a, set) or not isinstance(solution.set_b, set):
            return False

        # Check non-repeat
        all_selected_nodes = solution.set_a.union(solution.set_b)
        if len(all_selected_nodes) != len(solution.set_a) + len(solution.set_b):
            return False

        return True

    def get_observation(self) -> dict:
        return {
            "Selected Node": self.state_data["set_a_count"] + self.state_data["set_b_count"],
            "Cut Value": self.state_data["current_cut_value"],
        }

    def dump_result(self, dump_trajectory: bool=True, result_file: str="result.txt") -> str:
        content_dict = {
            "node_num": self.node_num,
            "set_a_count": self.state_data["set_a_count"],
            "set_b_count": self.state_data["set_b_count"]
        }
        content = super().dump_result(content_dict, dump_trajectory, result_file)
        return content
