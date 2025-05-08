import numpy as np
import tsplib95
import networkx as nx
import os
from aco import ACO


def load_data(data_path: str) -> None:
    try:
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        coordinates = problem.node_coords.values()  
        x = np.array([coord[0] for coord in coordinates])  
        y = np.array([coord[1] for coord in coordinates])  
        x_min, y_min = np.min(x), np.min(y)  
        x_max, y_max = np.max(x), np.max(y)  

        scale = max(x_max - x_min, y_max - y_min)
        distance_matrix /= scale
        return distance_matrix, scale
    except Exception as e:
        return None, None

def solve(distance_matrix, n_ants, n_iterations, heuristic):
    distance_matrix, scale = load_data(data_path)
    if distance_matrix is None:
        return None
    distance_matrix[np.diag_indices_from(distance_matrix)] = 1 # set diagonal to a large number
    heu = heuristic(distance_matrix.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(distance_matrix, heu, n_ants=n_ants)
    obj = aco.run(n_iterations)
    return obj.item() * scale

if __name__ == "__main__":
    from heuristic_base import heuristic_base
    from heuristic_reevo import heuristic_reevo
    heuristic = heuristic_base # [heuristic_base, heuristic_reevo]:

    n_ants = 30
    n_iterations = 100

    data_dir = os.path.join("..", "..", "output", "tsp", "data", "test_data")
    data_name = "kroA100.tsp"
    data_path = os.path.join(data_dir, data_name)
    
    result = solve(data_path, n_ants, n_iterations, heuristic)
    print(data_name, heuristic, result)