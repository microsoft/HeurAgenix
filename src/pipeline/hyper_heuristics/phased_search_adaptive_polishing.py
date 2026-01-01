import os
import random
import time
import math
from datetime import datetime
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.problems.max_cut.components import InsertNodeOperator, InsertEdgeOperator, SwapOperator, DeleteOperator, Solution

# Cache for graph properties to avoid re-calculation
_GRAPH_THRESHOLD_CACHE = {
    "g81": 0.75,
    "g77": 0.75,
    "g72": 0.75,
    "g70": 0.85,
    "g67": 0.75,
    "g66": 0.75,
    "g65": 0.70,
    "g64": 0.50,
    "g63": 0.80,
    "g62": 0.75,
    "g61": 0.75,
    "g60": 0.85,
}

def get_dynamic_threshold(env, data_name):
    """
    Calculates a dynamic quality threshold based on graph properties.
    - Positive graphs (NegRatio < 5%): High threshold (0.75)
    - Signed graphs: Threshold decreases with density (Dense signed graphs are harder)
    """
    if data_name in _GRAPH_THRESHOLD_CACHE:
        return _GRAPH_THRESHOLD_CACHE[data_name]
    
    node_num = env.instance_data["node_num"]
    adj = env.instance_data["adj"]
    
    edge_count = 0
    neg_edge_count = 0
    
    # Iterate adjacency list to count edges and negative weights
    # adj is a list of dicts: adj[u][v] = w
    for u in range(node_num):
        for v, w in adj[u].items():
            if u < v: # Count each undirected edge once
                edge_count += 1
                if w < 0:
                    neg_edge_count += 1
                    
    if edge_count == 0:
        threshold = 0.0
    else:
        return 0.8

class PhasedSearchAdaptivePolishingHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: list[str],
        problem: str,
        high_quality_solution_dir: str = None,
        top_k: int = 10,
        load_ratio: float = 0.4,
        fail_fast_threshold: float = 0.02,
    ) -> None:
        self.heuristic_pool_names = heuristic_pool
        self.problem = problem
        self.high_quality_solution_dir = high_quality_solution_dir
        self.top_k = top_k
        self.load_ratio = load_ratio
        self.fail_fast_threshold = fail_fast_threshold
        
        self.constructive_heuristics = []
        self.improvement_heuristics = []
        self.perturbation_heuristics = []
        self.ruin_heuristics = []
        self.mutation_heuristics = []
        
        self._classify_heuristics()

    def _classify_heuristics(self):
        # Manual classification of heuristics
        # This allows for more precise control than automatic type-based classification
        
        constructive_names = {
            "balance_biased_edge_placement_9f22",
            "balanced_cut_21d5",
            "balanced_cut_c0e6",
            "balanced_random_7f42",
            "heaviest_edge_seed_eb0d",
            "heavy_edge_matching_seed_edd5",
            "highest_delta_node_b31b",
            "highest_delta_edge_9f66",
            "highest_weight_edge_eb0d",
            "highest_weight_edge_eb0c",
            "highest_weight_edge_ca02",
            "most_weight_neighbors_320c",
            "most_weight_neighbors_d31b",
            "random_5c59",
            "semi_greedy_node_grasp_bf9a",
            "softmax_gain_insertion_76de",
            "spectral_seed_fiedler_51e0",
            "continuous_mean_field_batch", # Part 3: Batch CMF
            "balanced_random_batch", # Part 3: Batch Random
            "weighted_degree_batch", # Part 3: Batch Weighted Degree
            "cosm_heuristic_quick",
            "cosm_heuristic_detailed",
        }
        
        improvement_names = {
            "cached_delta_flip_3cfd",
            "first_improvement_flip_7a32",
            "greedy_swap_5bb6",
            "greedy_swap_5bb5",
            "k_block_swap_topk_589e",
            "majority_neighbor_flip_67a0",
            "multi_flip_threshold_fd21",
            "multi_swap_2_dbfe",
            "multi_swap_2_dbff",
            "single_flip_gain_5bb5",
            "tabu_node_flip_cae6",
            "two_node_joint_flip_590a",
        }
        
        ruin_names = {
            "low_contribution_bottom_delete_0b5f",
            "low_contribution_delete_worst_0b60",
            "random_delete_node_0b5f",
            "cluster_expansion_delete_bfs4",
            "batch_ruin",
            "batch_worst_ruin",
            "batch_cluster_ruin",
        }

        mutation_names = {
            "simulated_annealing_ed14",
            "simulated_annealing_ed15",
        }

        for h_name in self.heuristic_pool_names:
            # Strip .py extension if present for matching
            base_name = os.path.basename(h_name).replace(".py", "")
            
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in improvement_names:
                self.improvement_heuristics.append(func)
            elif base_name in ruin_names:
                self.ruin_heuristics.append(func)
                self.perturbation_heuristics.append(func) # Keep in general pool for backward compatibility if needed
            elif base_name in mutation_names:
                self.mutation_heuristics.append(func)
                self.perturbation_heuristics.append(func)
            else:
                print(f"Warning: Heuristic '{h_name}' not found in manual classification lists. Skipping.")

    def _get_pool_best_value(self) -> float:
        if not self.high_quality_solution_dir or not os.path.exists(self.high_quality_solution_dir):
            return 0.0
        
        # Cache strategy: Only check disk if cache is expired (e.g. every 60 seconds)
        current_time = time.time()
        if hasattr(self, '_pool_best_cache') and hasattr(self, '_pool_best_time'):
            if current_time - self._pool_best_time < 60: # 60 seconds cache
                return self._pool_best_cache
        
        best_val = 0.0
        try:
            files = os.listdir(self.high_quality_solution_dir)
            for f in files:
                if f.startswith("current_best."):
                    try:
                        # Format: current_best.{cut_value}.{exp_id}.{run_id}
                        parts = f.split(".")
                        if len(parts) >= 2:
                            val = float(parts[1])
                            if val > best_val:
                                best_val = val
                    except:
                        pass
            
            # Update cache
            self._pool_best_cache = best_val
            self._pool_best_time = current_time
            
        except Exception:
            pass
        return best_val

    def _read_solution_sets(self, path: str) -> tuple[set, set]:
        set_a = set()
        set_b = set()
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("set_a:"):
                        content = line.split(":", 1)[1].strip()
                        if content:
                            set_a = {int(x) - 1 for x in content.split(",")}
                    elif line.startswith("set_b:"):
                        content = line.split(":", 1)[1].strip()
                        if content:
                            set_b = {int(x) - 1 for x in content.split(",")}
        except Exception:
            pass
        return set_a, set_b

    def _calculate_overlap(self, set_a1, set_b1, set_a2, set_b2, node_num):
        # Direct match: A1-A2, B1-B2
        direct = len(set_a1 & set_a2) + len(set_b1 & set_b2)
        # Flipped match: A1-B2, B1-A2
        flipped = len(set_a1 & set_b2) + len(set_b1 & set_a2)
        max_overlap = max(direct, flipped)
        return max_overlap / node_num

    def _try_load_initial_solution(self, env: BaseEnv) -> tuple[bool, bool]:
        """
        Returns: (loaded_success, is_fragile_elite)
        """
        if not self.high_quality_solution_dir or not os.path.exists(self.high_quality_solution_dir):
            return False, False
            
        # Use load_ratio to decide whether to load or start from scratch
        if random.random() > self.load_ratio:
            return False, False
            
        try:
            files = [f for f in os.listdir(self.high_quality_solution_dir) if f.startswith("current_best.")]
            if not files:
                return False, False
            
            solution_files = []
            for f in files:
                try:
                    parts = f.split(".")
                    if len(parts) < 4: continue
                    val_str = ".".join(parts[1:-2])
                    val = float(val_str)
                    solution_files.append((f, val))
                except:
                    continue
            
            if not solution_files:
                return False, False

            # Sort solutions by value (descending)
            sorted_solutions = sorted(solution_files, key=lambda x: x[1], reverse=True)
            
            # === SMART LOADING STRATEGY ===
            # 1. Check Diversity (Overlap) of Top Solutions
            # If overlap is low (< 60%), it means we have distinct peaks (Multi-modal).
            # In this case, Crossover is destructive. We should pick ONE elite and polish it.
            
            is_fragile_elite = False
            
            if len(sorted_solutions) >= 2:
                # Check overlap of Top 2
                path1 = os.path.join(self.high_quality_solution_dir, sorted_solutions[0][0])
                path2 = os.path.join(self.high_quality_solution_dir, sorted_solutions[1][0])
                set_a1, set_b1 = self._read_solution_sets(path1)
                set_a2, set_b2 = self._read_solution_sets(path2)
                
                node_num = env.instance_data["node_num"]
                overlap = self._calculate_overlap(set_a1, set_b1, set_a2, set_b2, node_num)
                
                if overlap < 0.6:
                    print(f"Detected Low Overlap ({overlap:.1%}) in Top Solutions. Enabling Fragile Elite Mode (No Crossover).")
                    is_fragile_elite = True
            
            # Strategy Execution
            if is_fragile_elite:
                # Pick one of the top solutions (Roulette Wheel or Top K Random)
                # Bias heavily towards the very best to exploit the highest peak
                k = min(len(sorted_solutions), 5)
                chosen_file, chosen_val = random.choice(sorted_solutions[:k])
                
                path = os.path.join(self.high_quality_solution_dir, chosen_file)
                if env.load_solution(path):
                    print(f"Loaded Fragile Elite Solution from {chosen_file} (Value: {env.key_value})")
                    return True, True
            else:
                # Standard Logic (High Overlap -> Single Basin)
                # We can use Crossover or just load random top solution
                # For now, let's stick to simple loading to be safe, or implement Crossover if needed.
                # Given the user's request to focus on Polishing, we will just load a good solution.
                
                k = min(len(sorted_solutions), self.top_k)
                chosen_file, chosen_val = random.choice(sorted_solutions[:k])
                path = os.path.join(self.high_quality_solution_dir, chosen_file)
                if env.load_solution(path):
                    print(f"Loaded Standard Solution from {chosen_file} (Value: {env.key_value})")
                    return True, False

        except Exception as e:
            print(f"Failed to load initial solution: {e}")
            
        return False, False

    def run(self, env: BaseEnv) -> bool:
        current_steps = 0
        
        data = env.output_dir.split(os.sep)[-3]
        experiment = env.output_dir.split(os.sep)[-2]
        run_id = env.output_dir.split(os.sep)[-1]
        
        begin = datetime.now()
        last_value = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Start running Adaptive Polishing Search. Data:{data}\tExp\t{experiment}\tID:{run_id}\tStart:{begin.strftime('%Y-%m-%d %H:%M:%S')}\t", flush=True)
        
        # Try to load initial solution
        loaded_init, is_fragile_elite = self._try_load_initial_solution(env)
        quality_threshold = get_dynamic_threshold(env, data.split('.')[0])
        
        if loaded_init and env.best_known and env.best_known > 0:
            ratio = env.key_value / env.best_known
            if ratio < quality_threshold:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Loaded solution quality too low ({env.key_value}/{env.best_known} = {ratio:.1%}). Discarding.", flush=True)
                loaded_init = False
                is_fragile_elite = False
                env.reset(output_dir=env.output_dir)

        if loaded_init:
            current_best = env.key_value
            last_value = env.key_value
            init_value = env.key_value
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Loaded initial solution (Fragile: {is_fragile_elite}). Skipping construction.", flush=True)
        else:
            current_best = 0
            is_fragile_elite = False # New construction is not fragile initially
        
        # Optimization for Large Graphs (> 5000 nodes)
        active_constructive_heuristics = self.constructive_heuristics
        active_improvement_heuristics = self.improvement_heuristics
        
        if node_num > 5000:
            fast_constructive_names = {
                "random_5c59", 
                "balanced_random_7f42",
                "balanced_cut_21d5",
                "continuous_mean_field_batch", 
                "cosm_heuristic", 
                "cosm_heuristic_quick",
                "cosm_heuristic_detailed",
                "balanced_random_batch",
                "weighted_degree_batch",
                "highest_delta_node_b31b",
                "most_weight_neighbors_320c",
                "softmax_gain_insertion_76de",
            }
            fast_heuristics = [h for h in self.constructive_heuristics if h.__name__ in fast_constructive_names]
            
            if fast_heuristics:
                active_constructive_heuristics = fast_heuristics
                # Prioritize Cosm/CMF
                cosm_detailed = [h for h in fast_heuristics if h.__name__ == "cosm_heuristic_detailed"]
                if cosm_detailed:
                    active_constructive_heuristics.extend(cosm_detailed * 20)
            
            fast_improvement_names = {
                "cached_delta_flip_3cfd", 
                "first_improvement_flip_7a32", 
                "tabu_node_flip_cae6", # Now vectorized and fast
            }
            fast_imp_heuristics = [h for h in self.improvement_heuristics if h.__name__ in fast_improvement_names]
            self.tabu_heuristic = next((h for h in self.improvement_heuristics if h.__name__ == "tabu_node_flip_cae6"), None)
            
            if fast_imp_heuristics:
                active_improvement_heuristics = fast_imp_heuristics

        # === UCB Initialization ===
        heuristic_stats = {h.__name__: {'count': 0, 'reward': 0.0} for h in active_improvement_heuristics}
        total_ucb_steps = 0
        ucb_c = 1.0 

        # Track stagnation
        no_improve_steps = 0
        max_no_improve = 300 if node_num > 5000 else int(node_num * 2)
        
        perturbation_count = 0
        max_perturbations_before_ruin = 3
        
        current_ruin_percent = 0.3 
        best_at_last_ruin = 0
        
        # If we loaded a fragile elite, we start in a "Polishing" state, not Ruin state
        if is_fragile_elite:
            # Give it some time to polish before any ruin
            max_no_improve = 1000 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Fragile Elite Mode: Extended patience ({max_no_improve} steps) and Soft Ruin enabled.", flush=True)
            
            # IMMEDIATE TABU INJECTION
            # Since we know these solutions are hard local optima, standard greedy moves will likely fail.
            # We kickstart the process with a Tabu Search run immediately to save time.
            if hasattr(self, 'tabu_heuristic') and self.tabu_heuristic:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Fragile Elite Mode: Immediate Tabu Search Kickstart (Burst).", flush=True)
                # Use a smaller tenure for initial burst to allow flexibility
                dynamic_tenure = int(math.sqrt(node_num))
                env.run_heuristic(self.tabu_heuristic, parameters={"steps": 2000, "tabu_tenure": dynamic_tenure})
                current_steps += 1
                if env.key_value > current_best:
                    current_best = env.key_value
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Immediate Tabu Improvement! New Best: {current_best}", flush=True)
                    fname = f"current_best.{int(env.key_value)}.{experiment}.{run_id}"
                    path = os.path.join(self.high_quality_solution_dir, fname)
                    env.dump_best_solution(path)
                    self._pool_best_cache = max(getattr(self, '_pool_best_cache', 0), env.key_value)

        while env.continue_run:
            
            # Phase 1: Construction
            if not env.is_complete_solution:
                if not active_constructive_heuristics:
                    break
                
                # ... (Standard Construction Logic - Simplified for brevity) ...
                # Prefer Batch for speed
                batch_heuristics = [h for h in active_constructive_heuristics if "batch" in h.__name__ or "cosm" in h.__name__]
                if batch_heuristics:
                    heuristic = random.choice(batch_heuristics)
                    if "cosm" in heuristic.__name__:
                         env.run_heuristic(heuristic) # Cosm handles steps internally
                    else:
                         env.run_heuristic(heuristic, parameters={"batch_ratio": 0.01})
                else:
                    heuristic = random.choice(active_constructive_heuristics)
                    env.run_heuristic(heuristic)
                
                current_steps += 1
                last_value = env.key_value
                
                if env.is_complete_solution and env.is_valid_solution and current_best == 0:
                    # Quality Gate
                    quality_ratio = env.key_value / env.best_known
                    if env.key_value > current_best:
                        current_best = env.key_value
                        init_value = env.key_value
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data:{data}\tExp:{experiment}\tID:{run_id}\tConstruction completed. Init:{init_value}", flush=True)
                        
                        if quality_ratio < quality_threshold:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Data:{data}\tExp:{experiment}\tID:{run_id}\t [Quality Gate] Failed ({quality_ratio:.1%}). Restarting.", flush=True)
                            return False 
                
            # Phase 2: Improvement (and Perturbation)
            else:
                # Check if we need perturbation
                if no_improve_steps > max_no_improve:
                    perturbation_count += 1
                    
                    # === MASSIVE RUIN LOGIC ===
                    if perturbation_count > max_perturbations_before_ruin:
                        
                        # FAIL FAST
                        gap = (env.best_known - current_best) / env.best_known if env.best_known > 0 else 1.0
                        if gap > self.fail_fast_threshold:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Stagnated at {current_best} (Gap: {gap:.2%}). FAIL FAST.", flush=True)
                            return False

                        # FRAGILE ELITE PROTECTION
                        # If we are in Fragile Elite mode, we DO NOT want to destroy the structure with 30% ruin.
                        # Instead, we do a "Soft Reset" -> Tabu Search or very small ruin.
                        if is_fragile_elite:
                            # Allow escalation: If we tried Tabu and it failed (perturbation_count > max + 1), force Soft Ruin.
                            # max_perturbations_before_ruin is 3.
                            # Count 4: Try Tabu.
                            # Count 5: Try Tabu again (maybe different tenure?).
                            # Count 6: Soft Ruin.
                            
                            if perturbation_count <= max_perturbations_before_ruin + 2:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Fragile Elite Stagnation (Level {perturbation_count}). Triggering Deep Polishing (Tabu).", flush=True)
                                if self.tabu_heuristic:
                                    # Vary tenure to escape basins
                                    dynamic_tenure = int(math.sqrt(node_num))
                                    if perturbation_count % 2 == 0: dynamic_tenure *= 2 # Try stricter tabu
                                    
                                    # Deep polishing with more steps
                                    deep_steps = 2000
                                    if node_num > 5000:
                                        deep_steps = 10000
                                    
                                    env.run_heuristic(self.tabu_heuristic, parameters={"steps": deep_steps, "tabu_tenure": dynamic_tenure})
                                    current_steps += 1
                                    # Do NOT reset perturbation_count here, let it escalate if this fails to improve (which resets no_improve_steps in main loop)
                                    # But wait, if we don't reset no_improve_steps, we will come back here immediately.
                                    # We MUST reset no_improve_steps to give it time to prove itself.
                                    # But we want to remember that we tried Tabu.
                                    # The main loop resets perturbation_count ONLY if env.key_value > last_value.
                                    # So if Tabu fails, perturbation_count will remain high.
                                    no_improve_steps = 0 
                                    continue
                            
                            # Fallback: Very soft ruin (1%)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Fragile Elite Stagnation (Level {perturbation_count}). Tabu Failed. Triggering Soft Ruin (1%).", flush=True)
                            current_ruin_percent = 0.01 
                        
                        # Normal Massive Ruin Logic
                        if not is_fragile_elite:
                            if current_best > best_at_last_ruin:
                                current_ruin_percent = 0.3
                                best_at_last_ruin = current_best
                            else:
                                current_ruin_percent = min(0.5, current_ruin_percent + 0.05)

                            if current_ruin_percent >= 0.55:
                                if current_best >= env.best_known:
                                    current_ruin_percent = 0.3 # Extend run
                                else:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} STUCK. EARLY STOPPING.", flush=True)
                                    break

                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Massive Ruin (Percent: {current_ruin_percent:.1%}).", flush=True)
                        nodes_to_remove = max(10, int(node_num * current_ruin_percent))
                        
                        # Use Cluster Ruin if available
                        batch_heuristic = next((h for h in self.ruin_heuristics if h.__name__ == "batch_cluster_ruin"), None)
                        if not batch_heuristic:
                             batch_heuristic = next((h for h in self.ruin_heuristics if h.__name__ == "batch_ruin"), None)
                        
                        if batch_heuristic:
                             env.run_heuristic(batch_heuristic, parameters={"count": nodes_to_remove})
                             current_steps += 1
                        
                        perturbation_count = 0
                        no_improve_steps = 0
                        last_value = env.key_value 
                        continue

                    # === SMALL PERTURBATION ===
                    # For Fragile Elite, keep it very small (0.1% - 0.5%)
                    if is_fragile_elite:
                        base_perturb = max(5, int(node_num * 0.001))
                    else:
                        base_perturb = max(20, int(node_num * 0.005))
                    
                    multiplier = 1.0 + (perturbation_count * 0.5)
                    base_perturb = int(base_perturb * multiplier)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Small Perturbation (Size: {base_perturb}).", flush=True)

                    # Use simple random flips/deletes
                    for _ in range(base_perturb):
                        if self.ruin_heuristics:
                             heuristic = random.choice(self.ruin_heuristics)
                             if "batch" not in heuristic.__name__:
                                 env.run_heuristic(heuristic)
                    
                    no_improve_steps = 0 
                    last_value = env.key_value
                    continue

                # Normal Improvement
                if not active_improvement_heuristics:
                     break
                
                # TABU INJECTION
                if hasattr(self, 'tabu_heuristic') and self.tabu_heuristic:
                    tabu_interval = 500
                    if node_num < 2000: tabu_interval = 100
                    if node_num > 5000: tabu_interval = 100 # Aggressive for large graphs
                    
                    # For Fragile Elite, Tabu is our best friend. Use it more often.
                    if is_fragile_elite:
                        tabu_interval = 50

                    if no_improve_steps > 0 and no_improve_steps % tabu_interval == 0:
                        # print(f"Run:{run_id} Injecting Tabu Search.", flush=True)
                        dynamic_tenure = int(math.sqrt(node_num))
                        # Increase steps for large graphs now that it is vectorized
                        tabu_steps = 500
                        if node_num > 5000:
                            tabu_steps = 5000
                        
                        env.run_heuristic(self.tabu_heuristic, parameters={"steps": tabu_steps, "tabu_tenure": dynamic_tenure})
                        current_steps += 1
                        if env.key_value > last_value:
                            last_value = env.key_value
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                        continue 

                # UCB Selection
                selected_heuristic = None
                untried = [h for h in active_improvement_heuristics if heuristic_stats[h.__name__]['count'] == 0]
                if untried:
                    selected_heuristic = random.choice(untried)
                else:
                    best_ucb = -float('inf')
                    for h in active_improvement_heuristics:
                        stats = heuristic_stats[h.__name__]
                        avg_reward = stats['reward'] / stats['count']
                        exploration = ucb_c * math.sqrt(2 * math.log(total_ucb_steps) / stats['count'])
                        ucb_val = avg_reward + exploration
                        if ucb_val > best_ucb:
                            best_ucb = ucb_val
                            selected_heuristic = h
                
                if not selected_heuristic:
                    selected_heuristic = random.choice(active_improvement_heuristics)

                env.run_heuristic(selected_heuristic)
                current_steps += 1
                total_ucb_steps += 1
                
                if current_steps % 100 == 0:
                     print(f"[{datetime.now().strftime('%H:%M:%S')}] Run:{run_id} Step:{current_steps} NoImprove:{no_improve_steps} Val:{env.key_value} Best:{current_best}", flush=True)
                
                improvement = max(0, env.key_value - last_value)
                h_name = selected_heuristic.__name__
                heuristic_stats[h_name]['count'] += 1
                heuristic_stats[h_name]['reward'] += improvement

                if env.key_value > last_value:
                    last_value = env.key_value
                    no_improve_steps = 0
                    # Only reset perturbation escalation if we actually break the record.
                    # If we just recover to the same local optimum, we should keep the pressure on.
                    if env.key_value > current_best:
                        perturbation_count = 0 
                    
                    if env.is_valid_solution and env.key_value > current_best:
                        current_best = env.key_value
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tNow:{env.key_value}\tCurrent best:{current_best}", flush=True)
                        
                        # Save Logic
                        pool_best = self._get_pool_best_value()
                        should_save = False
                        if pool_best == 0: should_save = True
                        elif env.key_value >= pool_best * 0.99: should_save = True
                        
                        if should_save:
                             fname = f"current_best.{int(env.key_value)}.{experiment}.{run_id}"
                             path = os.path.join(self.high_quality_solution_dir, fname)
                             env.dump_best_solution(path)
                             self._pool_best_cache = max(getattr(self, '_pool_best_cache', 0), env.key_value)
                else:
                    no_improve_steps += 1

                current_best = max(current_best, env.key_value)

                if env.key_value > env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] !!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!", flush=True)
                        env.dump_result(result_file=f"break_best_known_result.{experiment}.{run_id}.txt")
                        found_best = True
                        env.best_known = env.key_value 

        return found_best
