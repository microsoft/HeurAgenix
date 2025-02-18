import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.online_bpp.components import Solution

class Env(BaseEnv):
    """Online bpp env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "online_bpp")
        self.capacity, self.item_num, self.item_sizes = self.data
        self.construction_steps = self.item_num
        self.key_item = "used_bin_num"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return self.current_solution.current_item == self.item_num

    def load_data(self, data_path: str) -> None:
        with open(data_path, "r") as file:
            first_line = file.readline().strip().split()
            capacity = int(first_line[0])
            item_num = int(first_line[1])
            item_sizes = []
            for _ in range(item_num):
                item_sizes.append(int(file.readline().strip()))
        return capacity, item_num, item_sizes

    def init_solution(self) -> None:
        return Solution([[] for _ in range(self.item_num)], 0)

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "capacity" (int): The capacity for each bin.
                - "item_num" (int): Total item number.
        """

        global_data_dict = {
            "capacity": self.capacity,
            "item_num": self.item_num,
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "current_item_size" (int): The size of current item to pack.
                - "used_bin_num" (int): The number of bins that has been used.
                - "used_capacity" (list[int]): List of used capacity for each bin.
                - "remaining_capacity" (list[int]): List of remaining capacity for each bin.
                - "num_items_in_box" (int): Total number of packed items.
                - "num_items_not_in_box" (int): Total number of unpacked items.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        current_item_size = None if solution.current_item >= len(self.item_sizes) else self.item_sizes[solution.current_item]
        used_bin_num = sum([1 for bin_index in range(self.item_num) if self.current_solution.sequences[bin_index] != []])
        used_capacity = [sum([self.item_sizes[item] for item in solution.sequences[bin_index]]) for bin_index in range(used_bin_num)]
        remaining_capacity = [self.capacity - used_capacity[bin_index] for bin_index in range(used_bin_num)]
        num_items_in_box = solution.current_item
        num_items_not_in_box = self.item_num - solution.current_item
        
        state_data_dict = {
            "current_solution": solution,
            "current_item_size": current_item_size,
            "used_bin_num": used_bin_num,
            "used_capacity": used_capacity,
            "remaining_capacity": remaining_capacity,
            "num_items_in_box": num_items_in_box,
            "num_items_not_in_box": num_items_not_in_box,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Bin Existence: Each bin in the solution must exist within the problem instance's range of bins.
            2. Item Existence: Each item in the solution must exist within the problem instance's range of items.
            3. Item Uniqueness: No item should appear more than once in the solution.
            4. Bin Capacity: The total size of items in each bin must not exceed the bin's capacity.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.sequences, list):
            return False
        # Check bin existence
        if len(solution.sequences) > self.item_num:
            return False

        item_set = set()
        for sequence in solution.sequences:
            # Check item existence
            if any(item < 0 or self.item_num < item for item in sequence):
                return False

            # Check item uniqueness
            if any(item in item_set for item in sequence):
                return False
            item_set = item_set.union(sequence)

            # Check bin capacity
            if sum([self.item_sizes[item] for item in sequence]) > self.capacity:
                return False
        return True

    def get_observation(self) -> dict:
        return {
            "Used Bin Num": self.state_data["used_bin_num"],
            "Num Items In Box": self.state_data["num_items_in_box"],
            "Num Items Not In Box": self.state_data["num_items_not_in_box"]
        }

    def dump_result(self, dump_trajectory: bool=True, result_file: str="result.txt") -> str:
        content_dict = {
            "item_num": self.item_num,
            "num_items_in_box": self.state_data["num_items_in_box"]
        }
        content = super().dump_result(content_dict, dump_trajectory, result_file)
        return content
