import numpy as np
import networkx as nx
import tsplib95
from src.problems.base.env import BaseEnv
from src.problems.tsp.components import Solution


class Env(BaseEnv):
    """TSP env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "tsp")
        self.node_num, self.distance_matrix = self.data
        self.construction_steps = self.node_num
        self.key_item = "current_cost"
        self.compare = lambda x, y: y - x

        # Precompute baseline cost once to avoid repeated heuristic runs
        self.baseline_cost = Env.nearest_neighbor_cost(self.distance_matrix)

    @property
    def is_complete_solution(self) -> bool:
        return len(set(self.current_solution.tour)) == self.node_num

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        node_num = len(distance_matrix)
        return node_num, distance_matrix

    def init_solution(self) -> None:
        return Solution(tour=[])

    def get_global_data(self) -> dict:
        return {
            "distance_matrix": self.distance_matrix,
            "node_num": self.node_num
        }

    def get_state_data(self, solution: Solution=None) -> dict:
        if solution is None:
            solution = self.current_solution

        visited_nodes = solution.tour
        unvisited_nodes = [node for node in range(self.node_num) if node not in visited_nodes]

        current_cost = sum(
            self.distance_matrix[visited_nodes[i]][visited_nodes[i+1]]
            for i in range(len(visited_nodes)-1)
        )
        if visited_nodes:
            current_cost += self.distance_matrix[visited_nodes[-1]][visited_nodes[0]]

        last_visited = visited_nodes[-1] if visited_nodes else None

        return {
            "current_solution": solution,
            "visited_nodes": visited_nodes,
            "visited_num": len(visited_nodes),
            "unvisited_nodes": unvisited_nodes,
            "unvisited_num": len(unvisited_nodes),
            "current_cost": current_cost,
            "last_visited": last_visited,
            "validation_solution": self.validation_solution
        }

    def validation_solution(self, solution: Solution=None) -> bool:
        node_set = set()
        if solution is None:
            solution = self.current_solution
        if not isinstance(solution, Solution) or not isinstance(solution.tour, list):
            return False
        for idx, node in enumerate(solution.tour):
            if not (0 <= node < self.node_num):
                return False
            if node in node_set:
                return False
            node_set.add(node)
            if idx < len(solution.tour) - 1:
                nxt = solution.tour[idx+1]
                if self.distance_matrix[node][nxt] == np.inf:
                    return False
        return True

    def get_observation(self) -> dict:
        return {
            "Visited Node": self.state_data["visited_num"],
            "Current Cost": self.state_data["current_cost"]
        }
    def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content_dict = {
            "node_num": self.node_num,
            "visited_num": self.state_data["visited_num"]
        }
        content = super().dump_result(content_dict=content_dict, dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
        return content

    @staticmethod
    def nearest_neighbor_cost(distance_matrix) -> float:
        """
        Compute the cost of a complete tour constructed using the nearest neighbor heuristic.
        """
        n = len(distance_matrix)
        current = 0
        unvisited = set(range(n)) - {current}
        tour = [current]
        while unvisited:
            nxt = min(unvisited, key=lambda j: distance_matrix[current][j])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        tour.append(tour[0])  # return to start
        return sum(
            distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)
        )

    def evaluate_quality(self, tour: list[int]=None, baseline_func=None) -> float:
        """
        Evaluate the relative quality of a given tour compared to a baseline.
        Quality is defined as the per-edge improvement ratio:
            quality = (baseline_avg_edge - current_avg_edge) / baseline_avg_edge
        Returns a value in [-1, 1], where positive means better than baseline.
        """
        if tour is None:
            tour = self.current_solution.tour
        # If default baseline, use precomputed cost
        if baseline_func is None or baseline_func == Env.nearest_neighbor_cost:
            baseline_cost = self.baseline_cost
        else:
            baseline_cost = baseline_func(self.distance_matrix)

        dm = self.distance_matrix
        n = dm.shape[0]
        visited = len(tour)

        # For trivial or empty path, consider max quality
        if visited < 5:
            return None

        # Compute cost of current tour including return to start
        current_cost = sum(
            dm[tour[i]][tour[i+1]] for i in range(visited-1)
        ) + dm[tour[-1]][tour[0]]

        # Average cost per edge
        baseline_avg = baseline_cost / n
        current_avg = current_cost / visited

        # Relative per-edge improvement
        quality = (baseline_avg - current_avg) / baseline_avg

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, quality))
