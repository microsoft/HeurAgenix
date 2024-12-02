import os
import random
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class RandomHyperHeuristic:
    def __init__(
        self,
        problem: str="tsp",
        heuristic_names: list[str]=None,
        heuristic_dir: str=None
    ) -> None:
        self.problem = problem
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        heuristic_names = os.listdir(self.heuristic_dir) if heuristic_names is None else heuristic_names
        heuristic_names = [heuristic_file.split(".")[0] for heuristic_file in heuristic_names]
        self.heuristic_pools = [load_heuristic(os.path.join(self.heuristic_dir, heuristic_name + ".py")) for heuristic_name in heuristic_names]

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        for _ in range(max_steps):
            heuristic = random.choice(self.heuristic_pools)
            _ = env.run_heuristic(heuristic)
        return env.is_complete_solution and env.is_valid_solution
