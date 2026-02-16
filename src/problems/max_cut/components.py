from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution for the MaxCut problem.
    Two sets of vertices representing the partition of the graph into two subsets.
    """
    def __init__(self, set_a: set[int], set_b: set[int], cut_value: float = None):
        self.set_a = set_a
        self.set_b = set_b
        self.cut_value = cut_value

    def __str__(self) -> str:
        # Output 1-based indexing for external compatibility
        set_a_str = ",".join([str(i + 1) for i in sorted(list(self.set_a))])
        set_b_str = ",".join([str(i + 1) for i in sorted(list(self.set_b))])
        return f"set_a: {set_a_str}\nset_b: {set_b_str}\n"


class InsertNodeOperator(BaseOperator):
    """Insert a node into one of the sets for the MaxCut solution."""
    def __init__(self, node: int, target_set: str):
        self.node = node
        self.target_set = target_set
        assert target_set in ["A", "B"]

class InsertEdgeOperator(BaseOperator):
    """ Insert an edge into the MaxCut solution with node_1 in set A and node_2 in set B."""
    def __init__(self, node_1: int, node_2: int):
        self.node_1 = node_1
        self.node_2 = node_2


class SwapOperator(BaseOperator):
    """Swap a list of nodes from origin set to the opposite set in the MaxCut solution."""
    def __init__(self, nodes: list[int]):
        self.nodes = nodes


class DeleteOperator(BaseOperator):
    """Delete a node from both sets in the MaxCut solution."""
    def __init__(self, node: int):
        self.node = node


class BatchDeleteOperator(BaseOperator):
    """Delete multiple nodes from the MaxCut solution in one go."""
    def __init__(self, nodes: list[int]):
        self.nodes = nodes


class BatchInsertNodeOperator(BaseOperator):
    """Insert multiple nodes into the MaxCut solution."""
    def __init__(self, nodes_to_a: list[int], nodes_to_b: list[int]):
        self.nodes_to_a = nodes_to_a
        self.nodes_to_b = nodes_to_b
