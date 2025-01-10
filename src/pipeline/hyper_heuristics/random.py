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
    ) -> None:
        self.heuristic_pools = [load_heuristic(heuristic, problem=problem) for heuristic in heuristic_pool]

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        current_steps = 0
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            _ = env.run_heuristic(heuristic)
            current_steps += 1
        return env.is_complete_solution and env.is_valid_solution
