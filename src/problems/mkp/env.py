import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.mkp.components import Solution

class Env(BaseEnv):
    """MKP env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "mkp")
        self.item_num, self.resource_num, self.profits, self.weights, self.capacities = self.data
        self.construction_steps = self.item_num
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
        return item_num, resource_num, profits, weights, capacities

    def init_solution(self) -> Solution:
        return Solution([False] * self.item_num)

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "item_num" (int): The total number of items available for selection.
                - "resource_num" (int): The number of resource dimensions or constraints.
                - "profits" (numpy.array): The profit value associated with each item.
                - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
                - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        """


        global_data_dict = {
            "item_num": self.item_num,
            "resource_num": self.resource_num,
            "profits": self.profits,
            "weights": self.weights,
            "capacities": self.capacities,
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "current_profit" (float): The total profit of the items included in the current solution.
                - "current_weights" (numpy.array): The total resource consumption for each dimension in the current solution.
                - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.
                - "items_in_knapsack" (list[int]): A list of item indices that are currently included in the knapsack.
                - "selected_item_num" (int): Number of items that have been selected.
                - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.
                - "unselected_item_num" (int): Number of items that have not been selected.
                - "feasible_items_to_add" (list[int]): A list of item indices that can be added to the knapsack without violating any resource constraints. 
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        current_weights = np.zeros(self.resource_num, dtype=float)
        current_profit = 0
        items_in_knapsack = []
        items_not_in_knapsack = []
        feasible_items_to_add = []

        for item_index, included in enumerate(solution.item_inclusion):
            if included:
                items_in_knapsack.append(item_index)
                current_profit += self.profits[item_index]
                current_weights += self.weights[:, item_index]
            else:
                items_not_in_knapsack.append(item_index)

        remaining_capacity = self.capacities - current_weights

        for item_index in items_not_in_knapsack:
            if np.all(remaining_capacity >= self.weights[:, item_index]):
                feasible_items_to_add.append(item_index)

        state_data_dict = {
            "current_solution": solution,
            "current_profit": current_profit,
            "current_weights": current_weights,
            "remaining_capacity": remaining_capacity,
            "items_in_knapsack": items_in_knapsack,
            "selected_item_num": len(items_in_knapsack),
            "items_not_in_knapsack": items_not_in_knapsack,
            "unselected_item_num": len(items_not_in_knapsack),
            "feasible_items_to_add": feasible_items_to_add,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

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
        if len(solution.item_inclusion) != self.item_num:
            return False

        # Check resource constraints
        total_weights = np.zeros(self.resource_num, dtype=float)
        for item_index, included in enumerate(solution.item_inclusion):
            if included:
                total_weights += self.weights[:, item_index]
        if np.any(total_weights > self.capacities):
            return False

        return True

    def get_observation(self) -> dict:
        return {
            "Item Inclusion Count": len(self.state_data["items_in_knapsack"]),
            "Profit": self.state_data["current_profit"],
        }

    def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content_dict = {
            "item_num": self.item_num,
            "resource_num": self.resource_num,
            "selected_item_num": self.state_data["items_in_knapsack"]
        }
        content = super().dump_result(content_dict=content_dict, dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
        return content