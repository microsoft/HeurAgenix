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
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                print(f"Run:{data}, {experiment}, {run_id}\tsteps:{current_steps}\tselected:{selected_nodes}\ttotal:{node_num}\tnow:{env.key_value}\tbest:{env.best_known}\ttime:{time_cost}", flush=True)
            if current_steps % 1000 == 0:
                if env.is_complete_solution and last_value == env.key_value:
                    print(f"Run:{data}, {experiment}, {run_id}\tsteps:{current_steps}\tselected:{selected_nodes}\ttotal:{node_num}\tnow:{env.key_value}\tbest:{env.best_known}\ttime:{time_cost}", flush=True)
                    print(f"No better solution found {last_value} => {env.key_value}, {current_steps}, stop", flush=True)
                    env.dump_result()
                    return found_best
                last_value = env.key_value
            _ = env.run_heuristic(heuristic)
            if env.key_value > env.best_known:
                print(f"Found best? try to check:{env.is_complete_solution} and {env.is_valid_solution}", flush=True)
                if env.is_complete_solution and env.is_valid_solution:
                    os.makedirs(env.output_dir)
                    print(f"break best from {env.best_known} to {env.key_value}, saved to {env.output_dir}", flush=True)
                    env.dump_result(result_file=f"break_best_known_result.txt")
                    found_best = True
                    return found_best
            current_steps += 1
            # env.dump_result()
        return found_best