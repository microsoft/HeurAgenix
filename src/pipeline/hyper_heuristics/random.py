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
        with open(os.path.join(env.output_dir, "running_log.txt"), "w") as f:
            f.write(f"current_steps\tselected_nodes\tnode_num\tcurrent_value\tbest_known\ttotal_time_cost(s)\n")
        begin = datetime.now()
        best_value = 0
        last_best = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        while current_steps <= max_steps and env.continue_run:
            heuristic = random.choice(self.heuristic_pools)
            _ = env.run_heuristic(heuristic)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                with open(os.path.join(env.output_dir, "running_log.txt"), "a") as f:
                    f.write(f"{current_steps}\t{selected_nodes}\t{node_num}\t{env.key_value}\t{env.best_known}\t{time_cost}\n")
                    if env.is_complete_solution and best_value == last_best:
                        f.write(f"No better result found.\n")
                        return env.is_complete_solution and env.is_valid_solution
                last_best = best_value
            if env.key_value >= env.best_known:
                env.dump_result(result_file=f"break_best_known_result.txt")
                found_best = True
            if env.is_complete_solution and env.key_value >= best_value:
                best_value = env.key_value
                env.dump_result(result_file=f"current_best_result.txt")
            current_steps += 1
        return found_best