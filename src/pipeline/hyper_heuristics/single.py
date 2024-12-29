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

    def run(self, env:BaseEnv, time_limitation: float=None, **kwargs) -> bool:
        heuristic_work = BaseOperator()
        while isinstance(heuristic_work, BaseOperator):
            heuristic_work = env.run_heuristic(self.heuristic)
        return env.is_valid_solution
