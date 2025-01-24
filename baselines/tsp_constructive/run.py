import os
import numpy as np
import tsplib95
import networkx as nx
from copy import copy

def load_data(data_path: str) -> None:
    try:
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        return distance_matrix
    except Exception as e:
        return None


def solve(dist_matrix, heuristic) -> float:
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    problem_size = dist_matrix.shape[0]
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = heuristic(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_matrix.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    cost = 0
    for i in range(problem_size):
        cost += dist_matrix[solution[i], solution[(i + 1) % problem_size]]
    return cost
    

if __name__ == '__main__':
    from heuristic_base import heuristic_base
    from heuristic_eoh import heuristic_eoh
    from heuristic_reevo import heuristic_reevo
    heuristic = heuristic_base # [heuristic_base, heuristic_eoh, heuristic_reevo]:

    data_dir = os.path.join("..", "..", "output", "tsp", "data", "test_data")
    data_name = "kroA100.tsp"
    data_path = os.path.join(data_dir, data_name)
    distance_matrix = load_data(data_path)

    if distance_matrix is not None:
        cost = solve(distance_matrix, heuristic)
        print(data_name, heuristic, cost)