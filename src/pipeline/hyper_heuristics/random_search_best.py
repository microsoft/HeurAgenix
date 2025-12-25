import datetime
import os
import random
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
        
    ) -> None:
        self.heuristic_pools = [load_function(heuristic, problem=problem) for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        current_steps = 0
        data = env.output_dir.split(os.sep)[-3]
        experiment = env.output_dir.split(os.sep)[-2]
        run_id = env.output_dir.split(os.sep)[-1]
        
        print(f"start running: {data}, {experiment}, {run_id}", flush=True)
            
        begin = datetime.now()
        last_value = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        current_best = 0
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                if env.is_complete_solution and last_value == env.key_value:
                    print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                    print(f"No better solution found {last_value} => {env.key_value}, {current_steps}, stop", flush=True)
                    env.dump_result()
                    return found_best
                last_value = env.key_value
            _ = env.run_heuristic(heuristic)
            if env.key_value == env.best_known:
                print(f"Found best known value at step {current_steps}", flush=True)

            # Logging
            current_best = max(current_best, env.key_value)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)

            if env.key_value == env.best_known:
                if env.is_complete_solution and env.is_valid_solution:
                    print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!")
                    env.dump_result(result_file=f"match_best_known_result.txt")
                    found_best = True
                    # Don't stop, try to improve more!
                    env.best_known = env.key_value # Update local best known to keep pushing

            # Check best known
            if env.key_value > env.best_known:
                if env.is_complete_solution and env.is_valid_solution:
                    print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!")
                    env.dump_result(result_file=f"break_best_known_result.txt")
                    found_best = True
                    # Don't stop, try to improve more!
                    env.best_known = env.key_value # Update local best known to keep pushing

            current_steps += 1
        return found_best