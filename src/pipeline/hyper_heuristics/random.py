import datetime
import os
import random
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from datetime import datetime

class RandomHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float=2.0,
    ) -> None:
        self.heuristic_pools = [load_function(heuristic, problem=problem) for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        current_steps = 0
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            begin = datetime.now()
            _ = env.run_heuristic(heuristic)
            end = datetime.now()
            time_cost = (end - begin).total_seconds() * 1000
            current_steps += 1
            print(current_steps, env.key_value, env.best_known, heuristic.__name__, time_cost)
            if env.key_value >= env.best_known:
                env.dump_result(result_file=f"found_best_best_result_{env.key_value}.txt")
        return env.is_complete_solution and env.is_valid_solution
