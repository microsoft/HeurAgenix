import os
import random
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class RandomHyperHeuristic:
    def __init__(
        self,
        problem: str="tsp",
        heuristic_dir: str=None
    ) -> None:
        self.problem = problem
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        heuristic_names = [heuristic_file.split(".")[0] for heuristic_file in os.listdir(self.heuristic_dir)]
        self.heuristic_pools = [load_heuristic(os.path.join(self.heuristic_dir, heuristic_name + ".py")) for heuristic_name in heuristic_names]

    def run(self, env:BaseEnv, max_steps: int=None, validation: bool=True, **kwargs) -> None:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        for index in range(max_steps):
            heuristic = random.choice(self.heuristic_pools)
            heuristic_works = env.run_heuristic(heuristic, validation=validation)
        return env.is_complete_solution
