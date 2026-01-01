import os
import random
from datetime import datetime
from src.problems.base.env import BaseEnv
from src.pipeline.hyper_heuristics.phased_search_adaptive_polishing import PhasedSearchAdaptivePolishingHyperHeuristic

class PhasedSearchBestKnownStartHyperHeuristic(PhasedSearchAdaptivePolishingHyperHeuristic):
    def __init__(self, heuristic_pool, problem, high_quality_solution_dir=None, top_k=10, load_ratio=1.0, fail_fast_threshold=0.02):
        # Force load_ratio to 1.0 to ensure we always try to load
        self.high_quality_solution_dir = high_quality_solution_dir
        super().__init__(heuristic_pool, problem, high_quality_solution_dir, top_k, 1.0, fail_fast_threshold)
        
    def _try_load_initial_solution(self, env: BaseEnv) -> tuple[bool, bool]:
        """
        Overrides the loading logic to specifically look for the best known solution file.
        Returns: (loaded_success, is_fragile_elite)
        """
        data_name = env.data_ref_name # e.g., "g81.mc"
        if data_name.endswith(".mc") or data_name.endswith(".vrp"):
            base_name = data_name.split(".")[0]
        else:
            base_name = data_name
        
        target_path = os.path.join(self.high_quality_solution_dir, f"best_known.txt")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Best Known from {target_path}...", flush=True)
        
        try:
            # Load using env.load_solution first
            if env.load_solution(target_path):
                # Check if set_b is missing and fix it
                node_num = env.instance_data["node_num"]
                all_nodes = set(range(node_num))
                
                if len(env.current_solution.set_b) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inferring set_b from set_a...", flush=True)
                    env.current_solution.set_b = all_nodes - env.current_solution.set_a
                    
                # Recalculate cut value just in case
                env.current_solution.cut_value = env.get_key_value(env.current_solution)
                env.problem_state = env.get_problem_state()
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded Best Known solution! Value: {env.key_value}", flush=True)
                
                # Force update best_known in env if our loaded solution is better (or equal)
                if env.key_value > env.best_known:
                    env.best_known = env.key_value
                    
                return True, True # True, True -> Loaded, Fragile Elite Mode
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] env.load_solution returned False.", flush=True)
                return False, False

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error loading best known: {e}", flush=True)
            return False, False
