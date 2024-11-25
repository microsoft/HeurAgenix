import os
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleHyperHeuristic:
    def __init__(
        self,
        heuristic_file: str,
        problem: str="tsp",
    ) -> None:
        self.heuristic = load_heuristic(heuristic_file, heuristic_dir=os.path.join("src", "problems", problem, "heuristics", "basic_heuristics"))

    def run(self, env:BaseEnv, time_limitation: float=10, **kwargs) -> bool:
        heuristic_works = True
        while not env.is_complete_solution:
            heuristic_works = env.run_heuristic(self.heuristic)
        return env.is_complete_solution and env.is_valid_solution