import os
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleConstructiveSingleImproveHyperHeuristic:
    def __init__(
        self,
        constructive_heuristic_file: str,
        improve_heuristic_file: str,
        problem: str="tsp"
    ) -> None:
        self.constructive_heuristic = load_heuristic(constructive_heuristic_file, heuristic_dir=os.path.join("src", "problems", problem, "heuristics", "basic_heuristics"))
        self.improve_heuristic = load_heuristic(improve_heuristic_file, heuristic_dir=os.path.join("src", "problems", problem, "heuristics", "basic_heuristics"))

    def run(self, env:BaseEnv, max_steps: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        heuristic_works = True
        while heuristic_works:
            heuristic_works = env.run_heuristic(self.constructive_heuristic)
        for _ in range(max_steps - env.construction_steps):
            heuristic_works = env.run_heuristic(self.improve_heuristic)
            if not heuristic_works:
                break
        return env.is_complete_solution and env.is_valid_solution