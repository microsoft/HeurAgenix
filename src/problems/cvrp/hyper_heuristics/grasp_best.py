import random
import os
from datetime import datetime
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_function

class GraspBestHyperHeuristic:
    """
    A GRASP-style (Greedy Randomized Adaptive Search Procedure) Hyper-Heuristic for CVRP.
    Phase 1: Randomized Construction (picking constructive heuristics until solution is complete).
    Phase 2: Local Search Optimization (repeatedly running improvement heuristics until local optima).
    """
    def __init__(self, heuristic_pool: list[str], problem: str, **kwargs) -> None:
        self.heuristic_names = heuristic_pool
        self.heuristics_dict = {}
        for h in heuristic_pool:
            self.heuristics_dict[h] = load_function(h, problem=problem)
        
        self.logger = kwargs.get("logger", None)
        self._classify_heuristics()

    def _classify_heuristics(self):
        constructive_names = {
            "farthest_insertion_4e1d",
            "greedy_f4c4",
            "nearest_neighbor_99ba",
            "random_bfdc",
            "min_cost_insertion_048f",
        }
        
        improvement_names = {
            "node_shift_between_routes_7b8a",
            "saving_algorithm_710e",
            "two_opt_0554",
            "three_opt_e8d7",
            "petal_algorithm_b384",
            "variable_neighborhood_search_614b",
        }
        
        self.constructive_pool = []
        self.improving_pool = []
        
        for h_name, h_func in self.heuristics_dict.items():
            # Strip file extension if somehow passed
            base_name = h_name.split(".")[0]
            if base_name in improvement_names:
                self.improving_pool.append(h_func)
            elif base_name in constructive_names:
                self.constructive_pool.append(h_func)
            else:
                # If unclassified, fallback to improvement to be safe
                self.improving_pool.append(h_func)
                
        # Failsafe
        if not self.constructive_pool:
            self.constructive_pool = list(self.heuristics_dict.values())
        if not self.improving_pool:
            self.improving_pool = list(self.heuristics_dict.values())

    def _log(self, msg):
        if self.logger:
            self.logger(msg)
        else:
            print(msg, flush=True)

    def run(self, env: BaseEnv) -> bool:
        if env.output_dir:
            data = env.output_dir.split(os.sep)[-3] if len(env.output_dir.split(os.sep)) >= 3 else "unknown"
            experiment = env.output_dir.split(os.sep)[-2] if len(env.output_dir.split(os.sep)) >= 2 else "unknown"
            run_id = env.output_dir.split(os.sep)[-1]
        else:
            data, experiment, run_id = "unknown", "unknown", "unknown"
        
        self._log(f"GraspBest running: Data={data}, Exp={experiment}, ID={run_id}")
            
        begin = datetime.now()
        
        if env.compare(100, 0) > 0:
            current_best = float('-inf')
        else:
            current_best = float('inf')
            
        epoch = 0
        total_steps = 0
        max_epochs = 1000000 
        
        while epoch < max_epochs and env.continue_run:
            env.reset() 
            
            # Phase 1: Construction 
            construction_attempts = 0
            while not env.is_complete_solution and env.continue_run:
                h = random.choice(self.constructive_pool)
                op = env.run_heuristic(h)
                total_steps += 1
                construction_attempts += 1
                if construction_attempts > env.instance_data['node_num'] * 10:
                    break
            
            if not env.is_complete_solution or not env.validation_solution():
                epoch += 1
                continue
                
            # Phase 2: Local Search 
            stagnation_count = 0
            while stagnation_count < len(self.improving_pool) and env.continue_run:
                h = self.improving_pool[stagnation_count]
                
                prev_cost = env.get_key_value()
                op = env.run_heuristic(h)
                total_steps += 1
                
                if isinstance(op, BaseOperator):
                    new_cost = env.get_key_value()
                    if env.compare(new_cost, prev_cost) > 0:
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1
                    
            # Phase 3: Evaluate against Best Known
            final_cost = env.get_key_value()
            if env.compare(final_cost, current_best) > 0:
                current_best = final_cost
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                self._log(f"[Epoch {epoch}] Data:{data} ID:{run_id} Step:{total_steps} Val:{current_best} BK:{env.best_known} Time:{time_cost/3600:.4f}h")
                
                if env.best_known is not None and env.compare(current_best, env.best_known) >= 0:
                    worker_name = getattr(self, "worker_id", run_id)
                    if env.compare(current_best, env.best_known) > 1e-3:
                        self._log(f"!!! BREAKTHROUGH FOUND: {current_best} (Better than {env.best_known}) !!!")
                        env.best_known = current_best
                        env.dump_result(result_file=f"breakthrough_from_worker_{worker_name}_{current_best}.txt")
                    else:
                        output_dir = env.output_dir
                        has_records = False
                        if output_dir and os.path.exists(output_dir):
                            for f in os.listdir(output_dir):
                                if f.startswith("breakthrough_") or f.startswith("match_"):
                                    has_records = True
                                    break
                        if not has_records:
                            env.dump_result(result_file=f"match_best_known_from_worker_{worker_name}_{current_best}.txt")
            
            epoch += 1

        return True
