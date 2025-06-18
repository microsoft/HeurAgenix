import os
import random
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic

class RandomHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float=2.0,
    ) -> None:
        self.heuristic_pools = [load_heuristic(heuristic, problem=problem) for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        current_steps = 0
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            _ = env.run_heuristic(heuristic)
            current_steps += 1
        return env.is_complete_solution and env.is_valid_solution
