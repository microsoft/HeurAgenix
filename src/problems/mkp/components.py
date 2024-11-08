from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution for the MKP.
    A list of boolean values where each boolean represents whether the corresponding item is included in the knapsack.
    """
    def __init__(self, item_inclusion: list[bool]):
        self.item_inclusion = item_inclusion

    def __str__(self) -> str:
        included_items = [str(item) for item, included in enumerate(self.item_inclusion) if included]
        return "selected_items: " + ",".join(included_items)


class ToggleOperator(BaseOperator):
    """Toggle the inclusion status of an item in the knapsack."""
    def __init__(self, item_index: int):
        self.item_index = item_index

    def run(self, solution: Solution) -> Solution:
        new_item_inclusion = solution.item_inclusion[:]
        new_item_inclusion[self.item_index] = not new_item_inclusion[self.item_index]
        return Solution(new_item_inclusion)

class AddOperator(BaseOperator):
    """Include an item in the knapsack."""
    def __init__(self, item_index: int):
        self.item_index = item_index

    def run(self, solution: Solution) -> Solution:
        new_item_inclusion = solution.item_inclusion[:]
        new_item_inclusion[self.item_index] = True
        return Solution(new_item_inclusion)

class RemoveOperator(BaseOperator):
    """Exclude an item from the knapsack."""
    def __init__(self, item_index: int):
        self.item_index = item_index

    def run(self, solution: Solution) -> Solution:
        new_item_inclusion = solution.item_inclusion[:]
        new_item_inclusion[self.item_index] = False
        return Solution(new_item_inclusion)

class SwapOperator(BaseOperator):
    """Swap the inclusion status of two items."""
    def __init__(self, item_index1: int, item_index2: int):
        self.item_index1 = item_index1
        self.item_index2 = item_index2

    def run(self, solution: Solution) -> Solution:
        new_item_inclusion = solution.item_inclusion[:]
        new_item_inclusion[self.item_index1], new_item_inclusion[self.item_index2] = \
        new_item_inclusion[self.item_index2], new_item_inclusion[self.item_index1]
        return Solution(new_item_inclusion)

class FlipBlockOperator(BaseOperator):
    """Flip the a list of items."""
    def __init__(self, index_list: list[int]):
        self.index_list = index_list

    def run(self, solution: Solution) -> Solution:
        new_item_inclusion = solution.item_inclusion[:]
        for i in self.index_list:
            new_item_inclusion[i] = not new_item_inclusion[i]
        return Solution(new_item_inclusion)
