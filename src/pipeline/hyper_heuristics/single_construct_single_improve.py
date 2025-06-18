import os
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleConstructiveSingleImproveHyperHeuristic:
    def __init__(
        self,
        constructive_heuristic_file: str,
        improve_heuristic_file: str,
        problem: str,
        iterations_scale_factor: float = 2.0
    ) -> None:
        self.constructive_heuristic = load_heuristic(constructive_heuristic_file, problem=problem)
        self.improve_heuristic = load_heuristic(improve_heuristic_file, problem=problem)
        self.iterations_scale_factor = iterations_scale_factor

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        heuristic_work = BaseOperator()
        while isinstance(heuristic_work, BaseOperator):
            heuristic_work = env.run_heuristic(self.constructive_heuristic)
        for _ in range(max_steps - env.construction_steps):
            heuristic_work = env.run_heuristic(self.improve_heuristic)
            if not heuristic_work:
                break
        return env.is_complete_solution and env.is_valid_solution