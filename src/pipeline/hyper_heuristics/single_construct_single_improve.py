import os
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleConstructiveSingleImproveHyperHeuristic:
    def __init__(
        self,
        constructive_heuristic_file: str,
        improve_heuristic_file: str,
        problem: str
    ) -> None:
        self.constructive_heuristic = load_heuristic(constructive_heuristic_file, problem=problem)
        self.improve_heuristic = load_heuristic(improve_heuristic_file, problem=problem)

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        heuristic_work = BaseOperator()
        while isinstance(heuristic_work, BaseOperator):
            heuristic_work = env.run_heuristic(self.constructive_heuristic)
        for _ in range(max_steps - env.construction_steps):
            heuristic_work = env.run_heuristic(self.improve_heuristic)
            if not heuristic_work:
                break
        return env.is_complete_solution and env.is_valid_solution