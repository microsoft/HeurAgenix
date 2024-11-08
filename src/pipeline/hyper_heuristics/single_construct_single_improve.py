import os
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleConstructiveSingleImproveHyperHeuristic:
    def __init__(
        self,
        constructive_heuristic_name: str,
        improve_heuristic_name: str,
        problem: str="tsp",
        heuristic_dir: str=None
    ) -> None:
        self.problem = problem
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        constructive_heuristic_file = constructive_heuristic_name if os.path.exists(constructive_heuristic_name) else os.path.join(self.heuristic_dir, constructive_heuristic_name + ".py")
        improve_heuristic_file = improve_heuristic_name if os.path.exists(improve_heuristic_name) else os.path.join(self.heuristic_dir, improve_heuristic_name + ".py")
        self.constructive_heuristic = load_heuristic(constructive_heuristic_file)
        self.improve_heuristic = load_heuristic(improve_heuristic_file)

    def run(self, env:BaseEnv, max_steps: int=None, validation: bool=True, **kwargs) -> None:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        heuristic_works = True
        while heuristic_works is not False:
            heuristic_works = env.run_heuristic(self.constructive_heuristic, validation=validation)
        for _ in range(max_steps - env.construction_steps):
            heuristic_works = env.run_heuristic(self.improve_heuristic, validation=validation)
            if heuristic_works is not True:
                break
        return env.is_complete_solution