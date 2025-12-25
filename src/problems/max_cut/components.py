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
        set_a_str = ",".join([str(i) for i in self.set_a])
        set_b_str = ",".join([str(i) for i in self.set_b])
        set_strings = f"set_a: {set_a_str}\nset_b: {set_b_str}\ncut_value: {self.cut_value}\n"
        return set_strings


class InsertNodeOperator(BaseOperator):
    """Insert a node into one of the sets for the MaxCut solution."""
    def __init__(self, node: int, target_set: str):
        self.node = node
        self.target_set = target_set
        assert target_set in ["A", "B"]
    
    def __str__(self):
        return f"InsertNodeOperator(node={self.node}, target_set='{self.target_set}')"

    def run(self, solution: Solution) -> Solution:
        new_set_a = set(solution.set_a)
        new_set_b = set(solution.set_b)
        if self.target_set == "A":
            assert self.node not in solution.set_b
            new_set_a.add(self.node)
        elif self.target_set == "B":
            assert self.node not in solution.set_a
            new_set_b.add(self.node)
        # Note: cut_value is not updated here, it should be updated by the Env or caller
        return Solution(new_set_a, new_set_b, solution.cut_value)


class InsertEdgeOperator(BaseOperator):
    """ Insert an edge into the MaxCut solution with node_1 in set A and node_2 in set B."""
    def __init__(self, node_1: int, node_2: int):
        self.node_1 = node_1
        self.node_2 = node_2

    def run(self, solution: Solution) -> Solution:
        new_set_a = set(solution.set_a)
        new_set_b = set(solution.set_b)
        assert self.node_1 not in solution.set_b
        assert self.node_2 not in solution.set_a
        new_set_a.add(self.node_1)
        new_set_b.add(self.node_2), solution.cut_value
        return Solution(new_set_a, new_set_b)


class SwapOperator(BaseOperator):
    """Swap a list of nodes from origin set to the opposite set in the MaxCut solution."""
    def __init__(self, nodes: list[int]):
        self.nodes = nodes

    def run(self, solution: Solution) -> Solution:
        new_set_a = set(solution.set_a)
        new_set_b = set(solution.set_b)
        for node in self.nodes:
            if node in solution.set_a:
                assert node not in solution.set_b
                new_set_a.remove(node)
                new_set_b.add(node)
            elif node in solution.set_b:
                assert node not in solution.set_a
                new_set_b.remove(node)
                new_set_a.add(node), solution.cut_value
        return Solution(new_set_a, new_set_b)


class DeleteOperator(BaseOperator):
    """Delete a node from both sets in the MaxCut solution."""
    def __init__(self, node: int):
        self.node = node

    def run(self, solution: Solution) -> Solution:
        new_set_a = set(solution.set_a)
        new_set_b = set(solution.set_b)
        if self.node in solution.set_a:
            new_set_a.remove(self.node)
        elif self.node in solution.set_b:
            new_set_b.remove(self.node)
        return Solution(new_set_a, new_set_b, solution.cut_value)


class BatchInsertNodeOperator(BaseOperator):
    """Insert multiple nodes into the MaxCut solution."""
    def __init__(self, nodes_to_a: list[int], nodes_to_b: list[int]):
        self.nodes_to_a = nodes_to_a
        self.nodes_to_b = nodes_to_b

    def __str__(self):
        return f"BatchInsertNodeOperator(nodes_to_a={len(self.nodes_to_a)}, nodes_to_b={len(self.nodes_to_b)})"

    def run(self, solution: Solution) -> Solution:
        new_set_a = set(solution.set_a)
        new_set_b = set(solution.set_b)
        
        new_set_a.update(self.nodes_to_a)
        new_set_b.update(self.nodes_to_b)
        
        return Solution(new_set_a, new_set_b, solution.cut_value)