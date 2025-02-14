import os
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleHyperHeuristic:
    def __init__(
        self,
        heuristic_file: str,
        problem: str,
    ) -> None:
        self.heuristic = load_heuristic(heuristic_file, problem=problem)

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        current_steps = 0
        heuristic_work = BaseOperator()
        while isinstance(heuristic_work, BaseOperator) and env.continue_run:
            heuristic_work = env.run_heuristic(self.heuristic)
            current_steps += 1
        return env.is_complete_solution and env.is_valid_solution
