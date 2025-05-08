import numpy as np

def heuristic_base(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    min_distance = np.inf
    for other in unvisited_nodes:
        if distance_matrix[current_node, other] < min_distance:
            min_distance = distance_matrix[current_node, other]
            next_node = other
    return next_node