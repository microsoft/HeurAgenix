import os
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.max_cut.components import Solution


class Env(BaseEnv):
    """MaxCut env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "max_cut")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "current_cut_value"
        self.compare = lambda x, y: x - y

    @property
    def is_complete_solution(self) -> bool:
        return len(self.current_solution.set_a) + len(self.current_solution.set_b) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> tuple:
        with open(data_path) as file:
            node_num = int(file.readline().split(" ", 1)[0])
            weight_matrix = np.zeros((node_num, node_num))
            for row in file:
                node_1, node_2, weight = [int(e) for e in row.strip("\n").split()]
                weight_matrix[node_1 - 1][node_2 - 1] = weight
                weight_matrix[node_2 - 1][node_1 - 1] = weight
        return {"node_num": node_num, "weight_matrix": weight_matrix}

    def init_solution(self) -> Solution:
        return Solution(set_a=set(), set_b=set())

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        current_cut_value = 0
        for node_a in solution.set_a:
            for node_b in solution.set_b:
                current_cut_value += self.instance_data["weight_matrix"][node_a][node_b]
        return current_cut_value

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
