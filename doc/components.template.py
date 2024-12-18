from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of xxx."""
    def __init__(self, **kwargs):
        # Init the solution.    
        pass

    def __str__(self) -> str:
        # str to print solution.
        pass


class Operator1(BaseOperator):
    def __init__(self, **kwargs):
        # Init the operator, but not run.
        pass

    def run(self, solution: Solution) -> Solution:
        # Run the operator and return a new solution.
        pass


class Operator2(BaseOperator):
    def __init__(self, **kwargs):
        # Init the operator, but not run.
        pass

    def run(self, solution: Solution) -> Solution:
        # Run the operator and return a new solution.
        pass

# This is an example from TSP.

# class Solution(BaseSolution):
#     def __init__(self, tour: list[int]):
#         self.tour = tour
#     def __str__(self) -> str:
#        return "tour: " + "->".join(map(str, self.tour + [self.tour[0]]))


# class AppendOperator(BaseOperator):
#     """Append the node at the end of the solution."""
#     def __init__(self, node: int):
#         self.node = node
#     def run(self, solution: Solution) -> Solution:
#         new_tour = solution.tour + [self.node]
#         return Solution(new_tour)


# class SwapOperator(BaseOperator):
#     """Swap two nodes in the solution. swap_node_pairs is a list of tuples, each containing the two nodes to swap."""
#     def __init__(self, swap_node_pairs: list[tuple[int, int]]):
#         self.swap_node_pairs = swap_node_pairs
#     def run(self, solution: Solution) -> Solution:
#         node_to_index = {node: index for index, node in enumerate(solution.tour)}
#         new_tour = solution.tour.copy()
#         for node_a, node_b in self.swap_node_pairs:
#             index_a = node_to_index.get(node_a)
#             index_b = node_to_index.get(node_b)
#             assert index_a is not None 
#             assert index_b is not None
#             new_tour[index_a], new_tour[index_b] = new_tour[index_b], new_tour[index_a]
#         return Solution(new_tour)

