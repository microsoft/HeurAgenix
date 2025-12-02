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
        run_id: int,
        iterations_scale_factor: float=2.0,
        
    ) -> None:
        self.heuristic_pools = [load_function(heuristic, problem=problem) for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor
        self.run_id = run_id

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        current_steps = 0
        #with open(os.path.join(env.output_dir, "running_log.txt"), "w") as f:
            # f.write(f"current_steps\tselected_nodes\tnode_num\tcurrent_value\tbest_known\ttotal_time_cost(s)\n")
        begin = datetime.now()
        best_value = 0
        last_best = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                print(f"run_id:{self.run_id}\tsteps:{current_steps}\tselected:{selected_nodes}\ttotal:{node_num}\tnow:{env.key_value}\tbest:{env.best_known}\ttime:{time_cost}\n")
                if env.is_complete_solution and best_value == last_best:
                    print(f"No better result found.\n")
                    return env.is_complete_solution and env.is_valid_solution
            _ = env.run_heuristic(heuristic)
            if current_steps > env.construction_steps and current_steps % 100 == 0:
                if last_best == env.key_value:
                    print("No better solution found, stop")
                    return found_best
                last_best = env.key_value
            if env.key_value >= env.best_known:
                if env.is_complete_solution and env.is_valid_solution:
                    os.makedirs(env.output_dir)
                    print(f"break best from {env.best_known} to {env.key_value}, saved to {env.output_dir}")
                    print(env.is_valid_solution)
                    env.dump_result(result_file=f"break_best_known_result.txt")
                    found_best = True
                    return found_best
            if env.is_complete_solution and env.key_value >= best_value:
                best_value = env.key_value
                # env.dump_result(result_file=f"current_best_result.txt")
            current_steps += 1
        return found_best