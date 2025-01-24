import numpy as np
import time
import os
import networkx as nx
import tsplib95

from gls_evol import guided_local_search, nearest_neighbor_2End, tour_cost_2End


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

def solve(data_path,time_limit, ite_max, perturbation_moves, heuristic):

    distance_matrix, scale = load_data(data_path)
    if distance_matrix is None:
        return None
    time.sleep(1)
    t = time.time()
    init_tour = nearest_neighbor_2End(distance_matrix, 0).astype(int)
    init_cost = tour_cost_2End(distance_matrix, init_tour)
    nb = 100
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:nb+1].astype(int)

    best_tour, best_cost, iter_i = guided_local_search(distance_matrix, nearest_indices, init_tour, init_cost,
                                                    t + time_limit, ite_max, perturbation_moves,
                                                    first_improvement=False, guide_algorithm=heuristic)
    return best_cost * scale


if __name__ == "__main__":
    from heuristic_base import heuristic_base
    from heuristic_eoh import heuristic_eoh
    from heuristic_reevo import heuristic_reevo
    heuristic = heuristic_base # [heuristic_base, heuristic_eoh, heuristic_reevo]:

    time_limit = 60 # maximum seconds for each instance
    ite_max = 1000 # maximum number of local search in GLS for each instance
    perturbation_moves = 1 # movers of each edge in each perturbation

    data_dir = os.path.join("..", "..", "output", "tsp", "data", "test_data")
    data_name = "kroA100.tsp"
    data_path = os.path.join(data_dir, data_name)
    distance_matrix, scale = load_data(data_path)

    result = solve(data_path, time_limit, ite_max, perturbation_moves, heuristic)
    print(data_name, heuristic.__name__, result * scale)