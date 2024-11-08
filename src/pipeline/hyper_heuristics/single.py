import os
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


class SingleHyperHeuristic:
    def __init__(
        self,
        heuristic_name: str,
        problem: str="tsp",
        heuristic_dir: str=None
    ) -> None:
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        heuristic_file = heuristic_name if os.path.exists(heuristic_name) else os.path.join(self.heuristic_dir, heuristic_name + ".py")
        self.heuristic = load_heuristic(heuristic_file)

    def run(self, env:BaseEnv, time_limitation: float=10, validation: bool=True, **kwargs) -> None:
        heuristic_works = True
        time_limitation = time_limitation * env.construction_steps
        while heuristic_works is not False:
            heuristic_works = env.run_heuristic(self.heuristic, validation=validation)
            if env.time_cost > time_limitation:
                return False
        return env.is_complete_solution