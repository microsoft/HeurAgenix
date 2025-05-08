import os
from aco import ACO
import numpy as np
from scipy.spatial import distance_matrix
import inspect
import tsplib95


def load_data(filepath):  
    try:
        problem = tsplib95.load(filepath)  
        capacity = problem.capacity
        node_pos = np.array([problem.node_coords[i] for i in problem.get_nodes()])  
        demand = np.array([problem.demands[i] for i in problem.get_nodes()])  
        return node_pos, demand, capacity
    except Exception as e:
        return None, None, None

def solve(data_path, n_ants, n_iterations, heuristic):
    node_pos, demand, capacity = load_data(data_path)
    if node_pos is None:
        return None
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristic(dist_mat.copy(), node_pos.copy(), demand.copy(), capacity) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, demand, heu, capacity, n_ants=n_ants)
    obj = aco.run(n_iterations)
    return obj.item()


if __name__ == "__main__":
    from heuristic_base import heuristic_base
    from heuristic_reevo import heuristic_reevo
    heuristic = heuristic_base # [heuristic_base, heuristic_reevo]:

    n_ants = 30
    n_iterations = 100

    data_dir = os.path.join("..", "..", "output", "cvrp", "data", "test_data")
    data_name = "A-n80-k10.vrp"
    data_path = os.path.join(data_dir, data_name)
    node_pos, demand, capacity = load_data(data_path)

    result = solve(data_path, n_ants, n_iterations, heuristic)
    print(data_name, heuristic.__name__, result)