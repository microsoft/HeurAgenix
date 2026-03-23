import random
import os
import time
from datetime import datetime
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_function

class BasicPhasedSearchHyperHeuristic:
    """
    Phase 1: Foundation Scheduling Infrastructure & Closed-Loop (Bootstrapping) for CVRP.
    Implements Phase 1 (Cold Start Construction) -> Phase 2 (VND Local Search until valley).
    Provides an initial single-machine evaluation and updating closed-loop.
    """
    def __init__(self, heuristic_pool, problem, worker_id=None, logger=None, **kwargs) -> None:
        self.heuristic_names = heuristic_pool
        self.problem = problem
        self.worker_id = str(worker_id)
        self.logger = logger
        
        # Load all available heuristic operators
        self.heuristics_dict = {}
        for h in heuristic_pool:
            self.heuristics_dict[h] = load_function(h, problem=problem)
            
        self._classify_heuristics()

    def _classify_heuristics(self):
        """
        Classifies heuristics in the pool based on predefined requirements:
        """
        # User defined constructive
        constructive_identifiers = [
            "nearest_neighbor",
            "greedy",
            "saving_algorithm"
        ]
        
        # User defined improvement
        improvement_identifiers = [
            "two_opt",
            "node_shift",
            "three_opt"
        ]
        
        self.constructives = []
        self.improvements = []
        
        for h_name, h_func in self.heuristics_dict.items():
            base_name = h_name.split(".")[0].lower()
            
            is_constructive = any(cid in base_name for cid in constructive_identifiers)
            is_improvement = any(iid in base_name for cid in improvement_identifiers for iid in improvement_identifiers)
            
            # Fallbacks
            if is_constructive:
                self.constructives.append((base_name, h_func))
            elif is_improvement:
                self.improvements.append((base_name, h_func))
            else:
                pass # Ignore out-of-plan operators or keep as backup
                
        # Extract functions for direct usage
        self.constructive_funcs = [func for _, func in self.constructives]
        self.improvement_funcs = [func for _, func in self.improvements]

        if not self.constructive_funcs:
            # Failsafe
            self.constructive_funcs = list(self.heuristics_dict.values())
        if not self.improvement_funcs:
            # Failsafe
            self.improvement_funcs = list(self.heuristics_dict.values())

    def _log(self, msg):
        if self.logger:
             self.logger(msg)
        else:
             print(msg, flush=True)

    def run(self, env: BaseEnv) -> bool:
        """Main entry point: Single-machine version for evaluation & update loop."""
        data = env.data_ref_name if hasattr(env, "data_ref_name") else "unknown"
        
        self._log(f"Bootstrapping BasicPhasedSearch initialized. Data={data}")
        
        begin = datetime.now()
        
        # In CVRP, key_value generally represents cost, where lower is better.

        current_best = float('inf')
        total_steps = 0
        epoch_count = 0
        max_epochs = 10000000 
        
        while epoch_count < max_epochs and env.continue_run:
            # [Phase 1: Cold Start Construction]
            env.reset()
            self._phase_1_construct(env)
            
            if not env.is_complete_solution or not env.validation_solution():
                epoch_count += 1
                continue
                
            # [Phase 2: VND Local Search to Exhaustive Limit]
            steps = self._phase_2_vnd_improve(env)
            total_steps += steps
            
            # [Evaluation]
            final_cost = env.get_key_value()
            if final_cost < current_best:
                current_best = final_cost
                
                # Report
                self._log(
                    f"Epoch={epoch_count}, Step={total_steps}, Val={current_best:.1f}, BK={env.best_known}"
                )
                
                if env.best_known is not None and current_best <= env.best_known:
                    if current_best < env.best_known:
                        self._log(f"!!! BREAKTHROUGH: {current_best} > {env.best_known} !!!")
                        env.best_known = current_best
                        env.dump_result(result_file=f"breakthrough_{self.worker_id}_{current_best:.4f}.txt")
                    else:
                        self._log(f"~~~ MATCHED BEST KNOWN: {current_best} ~~~")
                        env.dump_result(result_file=f"match_bk_{self.worker_id}_{current_best:.4f}.txt")
            
            epoch_count += 1

        return True

    def _phase_1_construct(self, env: BaseEnv):
        """Phase 1: Build from scratch until a complete feasible solution is found."""
        max_attempts = env.instance_data.get('node_num', 1000) * 10
        attempts = 0
        
        while not env.is_complete_solution and env.continue_run and attempts < max_attempts:
            # Randomly select a constructive operator
            h = random.choice(self.constructive_funcs)
            env.run_heuristic(h)
            attempts += 1
            
    def _phase_2_vnd_improve(self, env: BaseEnv) -> int:
        """Phase 2: Variable Neighborhood Descent (VND) Hill-climbing until all operators reach a valley."""
        steps = 0
        stagnation_count = 0
        
        # When all operators subsequently fail to produce any gain, we consider it has reached a true local optimum
        while stagnation_count < len(self.improvement_funcs) and env.continue_run:
            h = self.improvement_funcs[stagnation_count]
            
            prev_cost = env.get_key_value()
            op = env.run_heuristic(h)
            steps += 1
            
            if isinstance(op, BaseOperator):
                new_cost = env.get_key_value()
                # In CVRP, strictly decreasing Cost is an improvement
                if new_cost < prev_cost - 1e-5:
                    stagnation_count = 0  # Gain obtained, reset stagnation count
                else:
                    stagnation_count += 1 # No gain, switch to the next neighborhood operator
            else:
                stagnation_count += 1
                
        return steps
