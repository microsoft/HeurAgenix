import numpy as np

def heuristic_base(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    return 1 / distance_matrix