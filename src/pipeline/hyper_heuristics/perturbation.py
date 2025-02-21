import os
import random
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class PerturbationHyperHeuristic:
    def __init__(
        self,
        main_heuristic_file: str,
        perturbation_heuristic_file: str,
        problem: str,
        perturbation_ratio: float=0.1,
    ) -> None:
        self.main_heuristic = load_heuristic(main_heuristic_file, problem=problem)
        self.perturbation_heuristic = load_heuristic(perturbation_heuristic_file, problem=problem)
        self.perturbation_ratio = perturbation_ratio

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 3
        current_steps = 0
        heuristic_work = BaseOperator()
        while current_steps <= max_steps and isinstance(heuristic_work, BaseOperator) and env.continue_run:
            if random.random() < self.perturbation_ratio:
                heuristic = self.perturbation_heuristic
            else:
                heuristic = self.main_heuristic
            heuristic_work = env.run_heuristic(heuristic)
            current_steps += 1
        return env.is_complete_solution and env.is_valid_solution
