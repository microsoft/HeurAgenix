import os
import random
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class PerturbationHyperHeuristic:
    def __init__(
        self,
        main_heuristic: str,
        perturbation_heuristic: str,
        perturbation_ratio: float=0.1,
        problem: str="tsp",
        heuristic_dir: str=None
    ) -> None:
        self.problem = problem
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        main_heuristic_file = main_heuristic if os.path.exists(main_heuristic) else os.path.join(heuristic_dir, main_heuristic + ".py")
        perturbation_heuristic_file = perturbation_heuristic if os.path.exists(perturbation_heuristic) else os.path.join(heuristic_dir, perturbation_heuristic + ".py")
        self.main_heuristic = load_heuristic(main_heuristic_file)
        self.perturbation_heuristic = load_heuristic(perturbation_heuristic_file)
        self.perturbation_ratio = perturbation_ratio

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> None:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        heuristic_works = True
        while heuristic_works is not False:
            if random.random() < self.perturbation_ratio:
                heuristic = self.perturbation_heuristic
            else:
                heuristic = self.main_heuristic
            heuristic_works = env.run_heuristic(heuristic)
        return env.state_data
