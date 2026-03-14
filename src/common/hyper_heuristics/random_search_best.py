import datetime
import os
import random
import time
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from datetime import datetime

class RandomSearchBestHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float=2.0,
        **kwargs # Added to accept extra args from search_best.py
    ) -> None:
        self.heuristic_pools = [load_function(heuristic, problem=problem) for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor
        self.logger = kwargs.get("logger", None)

    def _log(self, msg):
        if self.logger:
            self.logger(msg)
        else:
            print(msg, flush=True)

    def run(self, env:BaseEnv) -> bool:
        # Random Search should run as long as the environment allows (Time-based usually)
        # We ignore iterations_scale_factor for "Search Best" scenario as we want to exhaust time budget.
        current_steps = 0
        
        if env.output_dir:
            data = env.output_dir.split(os.sep)[-3] if len(env.output_dir.split(os.sep)) >= 3 else "unknown"
            experiment = env.output_dir.split(os.sep)[-2] if len(env.output_dir.split(os.sep)) >= 2 else "unknown"
            run_id = env.output_dir.split(os.sep)[-1]
        else:
            data, experiment, run_id = "unknown", "unknown", "unknown"
        
        self._log(f"RandomSearchBest running: Data={data}, Exp={experiment}, ID={run_id}")
            
        begin = datetime.now()
        found_best = False

        if env.compare(100, 0) > 0:
            current_best = float('-inf')
        else:
            current_best = float('inf')
        
        # Determine strict stop limit if env doesn't enforce time
        max_steps = 1000000000 
        
        while current_steps <= max_steps and env.continue_run:
            if not self.heuristic_pools:
                break
                
            heuristic = random.choice(self.heuristic_pools)
            env.run_heuristic(heuristic)
            
            # Update Best
            
            if env.compare(env.key_value, current_best) > 0:
                if env.is_complete_solution and env.is_valid_solution:
                    current_best = env.key_value
                    self._log(f"Step:{current_steps} New Local Best: {current_best}")

            # Check Global Best Known
            if env.best_known is not None and env.compare(env.key_value, env.best_known) > 0:
                if env.is_complete_solution and env.is_valid_solution:
                    self._log(f"!!! NEW BEST FOUND: {env.key_value} (Better than {env.best_known}) !!!")
                    # Update local copy of best known to suppress repeated logs
                    env.best_known = env.key_value
                    env.dump_result(result_file=f"break_best_known_result_random_{run_id}.txt")
                    found_best = True

            # Logging
            if current_steps % 1000 == 0:
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                self._log(f"Data:{data} ID:{run_id} Step:{current_steps} Val:{env.key_value} Completed:{env.is_complete_solution} Current best:{current_best} BK:{env.best_known} Time:{time_cost/3600:.4f}h")

            current_steps += 1
            
        return found_best