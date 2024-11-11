import os
import random
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class RandomHyperHeuristic:
    def __init__(
        self,
        heuristic_dir: str=None,
        problem: str="tsp",
    ) -> None:
        heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        self.heuristic_pools = [load_heuristic(heuristic_name, heuristic_dir=heuristic_dir) for heuristic_name in os.listdir(heuristic_dir)]

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        for _ in range(max_steps):
            heuristic = random.choice(self.heuristic_pools)
            _ = env.run_heuristic(heuristic)
        return env.is_complete_solution and env.is_valid_solution
