import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.mkp.components import Solution

class Env(BaseEnv):
    """MKP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "mkp")
        self.construction_steps = self.instance_data["item_num"]
        self.key_item = "current_profit"
        self.compare = lambda x, y: x - y

    @property
    def is_complete_solution(self) -> bool:
        return True
    
    def load_data(self, data_path: str) -> tuple:
        with open(data_path, "r") as file:
            first_line = file.readline().strip().split()
            item_num = int(first_line[0])
            resource_num = int(first_line[1])
            profits = np.array(list(map(float, file.readline().strip().split())))
            weights = np.array([list(map(float, file.readline().strip().split())) for _ in range(resource_num)])
            capacities = np.array(list(map(float, file.readline().strip().split())))
        return {"item_num": item_num, "resource_num": resource_num, "profits": profits, "weights": weights, "capacities": capacities}

    def init_solution(self) -> Solution:
        return Solution([False] * self.instance_data["item_num"])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        current_profit = 0.0
        for item_index, included in enumerate(solution.item_inclusion):
            if included:
                current_profit += self.instance_data["profits"][item_index]
        return current_profit

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Solution length: The length of item_inclusion matches the total number of items.
            2. Resource constraints: The total weight of the selected items does not exceed the capacities for any of the resource dimensions.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.item_inclusion, list):
            return False

        # Check solution length
        if len(solution.item_inclusion) != self.instance_data["item_num"]:
            return False

        # Check resource constraints
        total_weights = np.zeros(self.instance_data["resource_num"], dtype=float)
        for item_index, included in enumerate(solution.item_inclusion):
            if included:
                total_weights += self.instance_data["weights"][:, item_index]
        if np.any(total_weights > self.instance_data["capacities"]):
            return False

        return True
