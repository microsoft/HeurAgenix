from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of TSP.
    A list of integers where each integer represents a node (city) in the TSP tour.
    The order of the nodes in the list defines the order in which the cities are visited in the tour.
    """
    def __init__(self, tour: list[int], node_num: int=None):
        self.tour = tour
        self.node_num = node_num

    def __str__(self) -> str:
        if len(self.tour) == self.node_num:
            return "tour: " + "->".join(map(str, self.tour + [self.tour[0]]))
        elif len(self.tour) > 0:
            return "tour(partial): " + "->".join(map(str, self.tour))
        else:
            return "tour: empty"


class AppendOperator(BaseOperator):
    """Append the node at the end of the solution."""
    def __init__(self, node: int):
        self.node = node

    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour + [self.node]
        return Solution(new_tour, solution.node_num)


class InsertOperator(BaseOperator):
    """Insert the node into the solution at the target position."""
    def __init__(self, node: int, position: int):
        self.node = node
        self.position = position

    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour[:self.position] + [self.node] + solution.tour[self.position:]
        return Solution(new_tour, solution.node_num)


class SwapOperator(BaseOperator):
    """Swap two nodes in the solution. swap_node_pairs is a list of tuples, each containing the two nodes to swap."""
    def __init__(self, swap_node_pairs: list[tuple[int, int]]):
        self.swap_node_pairs = swap_node_pairs

    def run(self, solution: Solution) -> Solution:
        node_to_index = {node: index for index, node in enumerate(solution.tour)}
        new_tour = solution.tour.copy()
        for node_a, node_b in self.swap_node_pairs:
            index_a = node_to_index.get(node_a)
            index_b = node_to_index.get(node_b)
            assert index_a is not None 
            assert index_b is not None
            new_tour[index_a], new_tour[index_b] = new_tour[index_b], new_tour[index_a]
        return Solution(new_tour, solution.node_num)


class ReplaceOperator(BaseOperator):
    """Replace a node with another one in the solution."""
    def __init__(self, node: int, new_node: int):
        self.node = node
        self.new_node = new_node

    def run(self, solution: Solution) -> Solution:
        index = solution.tour.index(self.node)
        new_tour = solution.tour[:index] + [self.new_node] + solution.tour[index + 1:]
        return Solution(new_tour, solution.node_num)


class ReverseSegmentOperator(BaseOperator):
    """Reverse multiple segments of indices in the solution."""
    def __init__(self, segments: list[tuple[int, int]]):
        self.segments = segments

    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour[:]

        for segment in self.segments:
            start_index, end_index = segment

            # Ensure the indices are within the bounds of the tour list
            assert 0 <= start_index < len(new_tour)
            assert 0 <= end_index < len(new_tour)

            # Ensure the start index comes before the end index
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            # Reverse the segment between start_index and end_index (inclusive)
            new_tour[start_index:end_index + 1] = reversed(new_tour[start_index:end_index + 1])

        return Solution(new_tour, solution.node_num)  