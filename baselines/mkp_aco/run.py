import os
import numpy as np
import torch
from aco import ACO


def load_data(data_path: str) -> tuple:
    try:
        with open(data_path, "r") as file:
            first_line = file.readline().strip().split()
            item_num = int(first_line[0])
            resource_num = int(first_line[1])
            profits = np.array(list(map(float, file.readline().strip().split())))
            weights = np.array([list(map(float, file.readline().strip().split())) for _ in range(resource_num)])
            capacities = np.array(list(map(float, file.readline().strip().split())))
            for i in range(weights.shape[0]):
                weights[i, :] = weights[i, :] / capacities[i]
        return profits, weights.T
    except Exception as e:
        return None, None


def solve(data_path, n_ants, n_iterations, heuristic):
    profits, weight = load_data(data_path)
    if profits is None:
        return None
    n, m = weight.shape
    heu = heuristic(profits.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(profits), torch.from_numpy(weight), torch.from_numpy(heu), n_ants)
    obj, _ = aco.run(n_iterations)
    return obj.item()

if __name__ == "__main__":
    from heuristic_base import heuristic_base
    from heuristic_reevo import heuristic_reevo
    heuristic = heuristic_base # [heuristic_base, heuristic_reevo]:

    n_ants = 30
    n_iterations = 100

    data_dir = os.path.join("..", "..", "output", "mkp", "data", "test_data")
    data_name = "mknap1_1.mkp"
    data_path = os.path.join(data_dir, data_name)

    result = solve(data_path, n_ants, n_iterations, heuristic)
    print(data_name, heuristic.__name__, result)