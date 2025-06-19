import numpy as np
import networkx as nx
import tsplib95
from src.problems.base.env import BaseEnv
from src.problems.tsp.components import Solution


class Env(BaseEnv):
    """TSP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "tsp")
        self.node_num, self.distance_matrix= self.instance_data["node_num"], self.instance_data["distance_matrix"]
        self.construction_steps = self.node_num
        self.key_item = "current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set(self.current_solution.tour)) == self.node_num

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        current_cost = sum([self.distance_matrix[solution.tour[index]][solution.tour[index + 1]] for index in range(len(solution.tour) - 1)])
        if len(solution.tour) > 0:
            current_cost += self.distance_matrix[solution.tour[-1]][solution.tour[0]]
        return current_cost

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        node_num = len(distance_matrix)
        return {"node_num": node_num, "distance_matrix": distance_matrix}

    def init_solution(self) -> None:
        return Solution(tour=[])

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node Existence: Each node in the solution must exist within the problem instance's range of nodes.
            2. Uniqueness: No node is repeated within the solution path, ensuring that each node is visited at most once.
            3. Connectivity: Each edge (from one node to the next) must be connected, i.e., not marked as infinite distance in the distance matrix.
        """
        node_set = set()
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.tour, list):
            return False
        if solution is not None and solution.tour is not None:
            for index, node in enumerate(solution.tour):
                # Check node existence
                if not (0 <= node < self.node_num):
                    return False

                # Check uniqueness
                if node in node_set:
                    return False
                node_set.add(node)

                # Check connectivity if not the last node
                if index < len(solution.tour) - 1:
                    next_node = solution.tour[index + 1]
                    if self.distance_matrix[node][next_node] == np.inf:
                        return False
        return True
