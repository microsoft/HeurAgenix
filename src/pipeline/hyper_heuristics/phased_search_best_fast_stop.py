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

class PhasedSearchFastStopBestHyperHeuristic:
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

    def _generate_backbone_solution(self, env: BaseEnv, top_k_files: list) -> bool:
        """
        Generates a solution based on the 'Backbone' of the population.
        Nodes with high consensus are fixed; uncertain nodes are randomized.
        """
        try:
            # 1. Load all top solutions
            solutions = []
            for f, val in top_k_files:
                path = os.path.join(self.high_quality_solution_dir, f)
                set_a, set_b = self._read_solution_sets(path)
                if set_a and set_b:
                    solutions.append({'a': set_a, 'b': set_b, 'val': val})
            
            if not solutions:
                return False

            # 2. Alignment (Handle Symmetry)
            # Reference is the best solution (first one)
            ref = solutions[0]
            aligned_solutions = [ref]
            
            for sol in solutions[1:]:
                # Calculate overlap with reference
                # Direct: A matches A, B matches B
                direct_match = len(sol['a'] & ref['a']) + len(sol['b'] & ref['b'])
                # Flipped: A matches B, B matches A
                flipped_match = len(sol['a'] & ref['b']) + len(sol['b'] & ref['a'])
                
                if flipped_match > direct_match:
                    # Flip this solution to align with reference
                    aligned_solutions.append({'a': sol['b'], 'b': sol['a'], 'val': sol['val']})
                else:
                    aligned_solutions.append(sol)
            
            # 3. Calculate Consensus
            node_num = env.instance_data["node_num"]
            counts_a = {i: 0 for i in range(node_num)}
            
            k = len(aligned_solutions)
            for sol in aligned_solutions:
                for node in sol['a']:
                    counts_a[node] += 1
            
            # 4. Construct Backbone
            new_set_a = set()
            new_set_b = set()
            
            # Thresholds for fixing
            # Strict backbone: 90% consensus
            # We can be slightly looser to encourage structure: 80%
            threshold_high = 0.8
            threshold_low = 0.2
            
            fixed_count = 0
            
            for node in range(node_num):
                prob_a = counts_a[node] / k
                
                if prob_a >= threshold_high:
                    new_set_a.add(node)
                    fixed_count += 1
                elif prob_a <= threshold_low:
                    new_set_b.add(node)
                    fixed_count += 1
                else:
                    # Uncertain / Unstable area -> Randomize
                    if random.random() < 0.5:
                        new_set_a.add(node)
                    else:
                        new_set_b.add(node)
            
            # Quality Gate: If consensus is too low, the backbone is essentially random.
            # We reject it to force a proper construction phase.
            consensus_ratio = fixed_count / node_num
            if consensus_ratio < 0.6:
                print(f"Backbone generation rejected: Consensus too low ({consensus_ratio:.1%}). Need > 60%.")
                return False

            print(f"Backbone Construction: Fixed {fixed_count}/{node_num} nodes ({fixed_count/node_num:.1%}). Randomizing rest.")
            
            # 5. Apply
            new_sol = Solution(new_set_a, new_set_b)
            env.current_solution = new_sol
            env.current_solution.cut_value = env.get_key_value(new_sol)
            env.problem_state = env.get_problem_state()
            
            return True
            
        except Exception as e:
            print(f"Backbone generation failed: {e}")
            return False

    def _try_load_initial_solution(self, env: BaseEnv) -> bool:
        if not self.high_quality_solution_dir or not os.path.exists(self.high_quality_solution_dir):
            return False
            
        # Use load_ratio to decide whether to load or start from scratch
        if random.random() > self.load_ratio:
            return False
            
        try:
            files = [f for f in os.listdir(self.high_quality_solution_dir) if f.startswith("current_best.")]
            if not files:
                print(f"Warning: No files found in {self.high_quality_solution_dir}", flush=True)
                return False
            
            # Simplified logic: Just pick from top K solutions found in the folder
            # No need to group by run_id, as random reconstruction will provide diversity
            solution_files = []
            
            for f in files:
                try:
                    parts = f.split(".")
                    # Format: current_best.{cut_value}.{exp_id}.{run_id}
                    # Value is everything between 'current_best.' and '.exp_id.run_id'
                    if len(parts) < 4:
                        continue
                        
                    val_str = ".".join(parts[1:-2])
                    val = float(val_str)
                    solution_files.append((f, val))
                except:
                    continue
            
            if not solution_files:
                print(f"Warning: Files found but none matched format 'current_best.VAL.EXP.ID' in {self.high_quality_solution_dir}. Example: {files[0]}", flush=True)
                return False

            # Sort solutions by value (descending)
            sorted_solutions = sorted(solution_files, key=lambda x: x[1], reverse=True)
            
            # === STRATEGY 1: BACKBONE EXTRACTION (Consensus) ===
            # If we have enough good solutions, try to extract the common structure
            # and randomize the unstable parts. This is very effective for large graphs.
            # Probability: 40%
            if len(sorted_solutions) >= 5 and random.random() < 0.4:
                # Use Top 10 for backbone
                top_k_files = sorted_solutions[:10]
                if self._generate_backbone_solution(env, top_k_files):
                    print(f"Successfully generated Backbone Solution from Top {len(top_k_files)} (Value: {env.key_value})")
                    return True

            # === STRATEGY 2: CROSSOVER (Hybridization) ===
            # With 50% probability (of the remaining 60%), if we have enough parents, create a hybrid child.
            # This combines traits from two high-quality solutions to explore new basins.
            if len(sorted_solutions) >= 2 and random.random() < 0.5:
                # Select two distinct parents from Top K
                k = min(len(sorted_solutions), self.top_k)
                parent1_file, _ = random.choice(sorted_solutions[:k])
                parent2_file, _ = random.choice(sorted_solutions[:k])
                
                # Try to get a different second parent
                attempts = 0
                while parent1_file == parent2_file and attempts < 5:
                    parent2_file, _ = random.choice(sorted_solutions[:k])
                    attempts += 1
                
                if parent1_file != parent2_file:
                    path1 = os.path.join(self.high_quality_solution_dir, parent1_file)
                    path2 = os.path.join(self.high_quality_solution_dir, parent2_file)
                    
                    set_a1, set_b1 = self._read_solution_sets(path1)
                    set_a2, set_b2 = self._read_solution_sets(path2)
                    
                    if set_a1 and set_a2:
                        # SYMMETRY FIX: MaxCut solutions are symmetric (A, B) == (B, A).
                        # Align Parent 2 to Parent 1 to maximize overlap.
                        overlap_direct = len(set_a1 & set_a2)
                        overlap_flipped = len(set_a1 & set_b2)
                        
                        if overlap_flipped > overlap_direct:
                            # Flip Parent 2
                            set_a2, set_b2 = set_b2, set_a2
                            max_overlap = overlap_flipped
                        else:
                            max_overlap = overlap_direct
                        
                        # Quality Gate for Crossover:
                        # If parents are too different (low overlap), the child will be mostly random noise.
                        # We reject such pairs to avoid polluting the search with bad seeds.
                        node_num = env.instance_data["node_num"]
                        overlap_ratio = max_overlap / node_num
                        if overlap_ratio < 0.6:
                            # print(f"Crossover rejected: Parents too different (Overlap: {overlap_ratio:.1%}). Need > 60%.")
                            pass # Silently skip to try other strategies
                        else:
                            # Crossover Logic:
                            # 1. Intersection: Keep nodes that agree
                            # 2. Disagreement: Randomly assign
                            new_set_a = set()
                            new_set_b = set()
                            
                            # Union of all nodes involved (should be all nodes if complete)
                            all_nodes = set_a1 | set_b1 | set_a2 | set_b2
                            
                            for node in all_nodes:
                                in_a1 = node in set_a1
                                in_a2 = node in set_a2
                                
                                if in_a1 and in_a2:
                                    new_set_a.add(node)
                                elif not in_a1 and not in_a2:
                                    new_set_b.add(node)
                                else:
                                    # Disagreement
                                    if random.random() < 0.5:
                                        new_set_a.add(node)
                                    else:
                                        new_set_b.add(node)
                            
                            # Create and set solution
                            new_sol = Solution(new_set_a, new_set_b)
                            env.current_solution = new_sol
                            # Recalculate value
                            env.current_solution.cut_value = env.get_key_value(new_sol)
                            env.problem_state = env.get_problem_state()
                            
                            print(f"Successfully generated Hybrid Solution from {parent1_file} and {parent2_file} (Value: {env.key_value})")
                            return True

            # === STRATEGY 3: DIVERSITY INJECTION (Selection) ===
            # Instead of always picking the absolute best, we pick from a wider range (Top 20)
            # to avoid getting stuck in the same local optimum basin.
            # We also give a small chance to pick a random "good" solution from the pool.
            
            if random.random() < 0.3:
                # 30% chance to pick completely random from the pool (Exploration)
                chosen_file, chosen_val = random.choice(sorted_solutions)
            else:
                # 70% chance to pick from Top K (Exploitation)
                k = min(len(sorted_solutions), self.top_k)
                chosen_file, chosen_val = random.choice(sorted_solutions[:k])
            
            path = os.path.join(self.high_quality_solution_dir, chosen_file)
            if env.load_solution(path):
                print(f"Successfully loaded initial solution from {chosen_file} (Value: {env.key_value})")
                return True
        except Exception as e:
            print(f"Failed to load initial solution: {e}")
            
        return False

    def run(self, env: BaseEnv) -> bool:
        current_steps = 0
        
        data = env.output_dir.split(os.sep)[-3]
        experiment = env.output_dir.split(os.sep)[-2]
        run_id = env.output_dir.split(os.sep)[-1]
        
        begin = datetime.now()
        last_value = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        print(f"Start running phased search. Data:{data}\tExp\t{experiment}\tID:{run_id}\tStart:{begin.strftime('%Y-%m-%d %H:%M:%S')}\t", flush=True)
        
        # Try to load initial solution
        loaded_init = self._try_load_initial_solution(env)
        quality_threshold = get_dynamic_threshold(env, data.split('.')[0])
        
        # Quality Gate for Initial Solution
        # If the loaded solution is significantly worse than best known (e.g. < 60%), discard it.
        # This prevents starting from "random-like" backbones or bad seeds.
        if loaded_init and env.best_known and env.best_known > 0:
            ratio = env.key_value / env.best_known
            if ratio < quality_threshold:
                print(f"Run:{run_id} Loaded solution quality too low ({env.key_value}/{env.best_known} = {ratio:.1%}). Discarding and restarting construction.", flush=True)
                loaded_init = False
                env.reset(output_dir=env.output_dir)

        if loaded_init:
            current_best = env.key_value
            last_value = env.key_value
            init_value = env.key_value
            print(f"Run:{run_id} Loaded initial solution with value {current_best}. Skipping construction.", flush=True)
        else:
            current_best = 0
        
        # Optimization for Large Graphs (> 5000 nodes):
        # Use only O(1) or O(N) constructive heuristics to avoid O(N^2) bottlenecks.
        # This allows the search to reach the improvement/ruin phase much faster.
        active_constructive_heuristics = self.constructive_heuristics
        active_improvement_heuristics = self.improvement_heuristics
        
        if node_num > 5000:
            fast_constructive_names = {
                "random_5c59", 
                "balanced_random_7f42",
                "balanced_cut_21d5",
                "continuous_mean_field_batch", # New physics-inspired heuristic
                "cosm_heuristic", # CPU Optimized Cosm
                "cosm_heuristic_quick",
                "cosm_heuristic_detailed",
                "balanced_random_batch",
                "weighted_degree_batch",
                "highest_delta_node_b31b", # Include slow ones for hybrid strategy?
                "most_weight_neighbors_320c",
                "softmax_gain_insertion_76de",
            }
            # Filter heuristics by name
            fast_heuristics = [h for h in self.constructive_heuristics if h.__name__ in fast_constructive_names]
            
            if fast_heuristics:
                print(f"Large graph detected ({node_num} nodes). Switching to Hybrid Constructive Heuristics.", flush=True)
                active_constructive_heuristics = fast_heuristics
                
                # Prioritize Cosm/CMF if available
                cosm_heuristic = [h for h in fast_heuristics if h.__name__ == "cosm_heuristic"]
                cosm_quick = [h for h in fast_heuristics if h.__name__ == "cosm_heuristic_quick"]
                cosm_detailed = [h for h in fast_heuristics if h.__name__ == "cosm_heuristic_detailed"]
                cmf_heuristic = [h for h in fast_heuristics if h.__name__ == "continuous_mean_field_batch"]
                
                if cosm_quick or cosm_detailed:
                    # Prioritize the new split heuristics
                    if cosm_quick:
                        active_constructive_heuristics.extend(cosm_quick * 2)
                    if cosm_detailed:
                        active_constructive_heuristics.extend(cosm_detailed * 20)
                elif cosm_heuristic:
                    # Give Cosm a much higher weight (Primary Choice)
                    active_constructive_heuristics.extend(cosm_heuristic * 10)
                elif cmf_heuristic:
                    active_constructive_heuristics.extend(cmf_heuristic * 5)
            else:
                print("Warning: Large graph detected but no fast heuristics found. Using default pool.")

            # Also filter IMPROVEMENT heuristics for large graphs
            # We keep 'cached_delta_flip' for speed (it replaces slow greedy swaps).
            # We REMOVE 'tabu_node_flip' from the random pool because it is O(N^2) and too slow for frequent use.
            # Instead, we will trigger it conditionally when stuck.
            fast_improvement_names = {
                "cached_delta_flip_3cfd", # O(1) update, extremely fast greedy descent
                "first_improvement_flip_7a32", # O(N) scan, good for diversity
            }
            fast_imp_heuristics = [h for h in self.improvement_heuristics if h.__name__ in fast_improvement_names]
            
            # Extract Tabu for special use
            self.tabu_heuristic = next((h for h in self.improvement_heuristics if h.__name__ == "tabu_node_flip_cae6"), None)
            
            if fast_imp_heuristics:
                print(f"Large graph detected. Using optimized improvement set (Speed only): {[h.__name__ for h in fast_imp_heuristics]}")
                print(f"Tabu heuristic '{self.tabu_heuristic.__name__ if self.tabu_heuristic else 'None'}' reserved for stagnation handling.")
                active_improvement_heuristics = fast_imp_heuristics
            else:
                print("Warning: No optimized improvement heuristics found! Using full pool.")
            self.tabu_heuristic = next((h for h in self.improvement_heuristics if h.__name__ == "tabu_node_flip_cae6"), None)

        # === UCB Initialization ===
        # Track usage and rewards for Multi-Armed Bandit strategy
        heuristic_stats = {h.__name__: {'count': 0, 'reward': 0.0} for h in active_improvement_heuristics}
        total_ucb_steps = 0
        ucb_c = 1.0 # Exploration constant

        # Track stagnation
        no_improve_steps = 0
        max_no_improve = int(node_num * 2)  # Dynamic threshold based on problem size
        
        # Optimization for Large Graphs (> 5000 nodes):
        if node_num > 5000:
            # Increase patience for large graphs as operators might be slower but more impactful
            # Or decrease it if we want more frequent perturbations. 
            # For Tabu-like behavior, we want to explore local optima fully.
            # FIX: 4*N is too long for 20k nodes (80k steps ~ 40 hours). 
            # We need to fail fast and ruin often.
            # UPDATE: Aggressive Fail Fast Strategy (300 steps ~ 20 mins stagnation)
            max_no_improve = 300 

        
        # Track perturbation cycles for massive ruin (Large Neighborhood Search)
        perturbation_count = 0
        max_perturbations_before_ruin = 3 # Fail fast: Trigger massive ruin sooner
        
        # Adaptive Ruin Parameters
        current_ruin_percent = 0.3 # Start with stronger ruin (30%) to escape deep valleys
        best_at_last_ruin = 0
        
        # Polishing State
        polishing_attempted = False
        
        # We rely on the Early Stopping mechanism (stagnation at max ruin) to terminate the run.
        # This allows the search to continue as long as it is making progress.
        while env.continue_run:
            
            # Phase 1: Construction
            if not env.is_complete_solution:
                if not active_constructive_heuristics:
                    print("Error: No constructive heuristics available but solution is incomplete.")
                    break
                
                # Hierarchical Hybrid Construction Strategy
                # Mix Fast Batch Heuristics (High Prob) with Slow Precision Heuristics (Low Prob)
                
                # 1. Identify Batch vs Single Heuristics
                # We want to use it sparingly (20% prob), not frequently (80% prob).
                batch_heuristics = [h for h in active_constructive_heuristics if "batch" in h.__name__]
                
                # Treat Cosm Detailed as a batch heuristic because it constructs the full solution efficiently
                cosm_detailed_list = [h for h in active_constructive_heuristics if h.__name__ == "cosm_heuristic_detailed"]
                if cosm_detailed_list:
                    batch_heuristics.extend(cosm_detailed_list)

                single_heuristics = [h for h in active_constructive_heuristics if h not in batch_heuristics]
                
                # 2. Determine Strategy
                use_batch = False
                # Prefer batch if available, with 70% probability (Balanced Hybrid)
                # Lower probability (e.g. 0.7 vs 0.9) increases diversity by allowing more random/greedy single insertions.
                if batch_heuristics and (not single_heuristics or random.random() < 0.5):
                    use_batch = True
                
                # SMART REBUILDING: If we are rebuilding (current_best > 0), avoid random heuristics.
                # We want to repair the solution with high-quality moves, not random noise.
                is_rebuilding = current_best > 0
                if is_rebuilding:
                    # Filter out random heuristics
                    # Keep: CMF, Weighted Degree, Softmax Gain, etc.
                    # Remove: Balanced Random, Random
                    smart_batch = [h for h in batch_heuristics if "random" not in h.__name__]
                    smart_single = [h for h in single_heuristics if "random" not in h.__name__]
                    
                    # For small graphs, we might need some randomness to escape local optima
                    # FIX: Allow randomness with 50% probability even for large graphs to avoid "Ruin & Recreate Trap"
                    if node_num < 2000 or random.random() < 0.5:
                         # Keep some random heuristics but prioritize smart ones?
                         # Or just disable this filter for small graphs.
                         # Let's disable the filter for small graphs to allow diversity.
                         pass
                    else:
                        if smart_batch:
                            batch_heuristics = smart_batch
                            use_batch = True # Prefer batch for speed if smart ones exist
                        elif smart_single:
                            single_heuristics = smart_single
                            use_batch = False
                    # If no smart heuristics found (unlikely), fall back to whatever we have
                
                # 3. Execute
                if use_batch:
                    heuristic = random.choice(batch_heuristics)
                    
                    # Special handling for Cosm Heuristic (Quick vs Slow)
                    if heuristic.__name__ == "cosm_heuristic":
                        # Randomize steps for diversity: 100 (fast) to 500 (precise)
                        # This creates diverse starting points in different basins
                        steps = random.randint(100, 500)
                        env.run_heuristic(heuristic, parameters={"steps": steps})
                    elif heuristic.__name__ in ["cosm_heuristic_quick", "cosm_heuristic_detailed"]:
                        # New split heuristics handle steps internally (dynamic based on graph size)
                        env.run_heuristic(heuristic)
                    else:
                        # Use ratio instead of fixed batch size
                        # 1% of nodes per batch allows for ~100 phases of construction (Fine-grained)
                        # Adaptive Batch Size: Smaller batches for small graphs or rebuilding
                        batch_ratio = 0.01
                        if node_num < 2000 or is_rebuilding:
                            batch_ratio = 0.01 # More precise construction
                            
                        env.run_heuristic(heuristic, parameters={"batch_ratio": batch_ratio})
                else:
                    # Single Insertion (Precision)
                    if single_heuristics:
                        heuristic = random.choice(single_heuristics)
                        env.run_heuristic(heuristic)
                    elif batch_heuristics:
                        # Fallback: Use batch heuristic as single insertion
                        heuristic = random.choice(batch_heuristics)
                        env.run_heuristic(heuristic)
                
                current_steps += 1
                
                last_value = env.key_value
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                
                # Check if construction just finished
                if env.is_complete_solution and env.is_valid_solution and current_best == 0:
                    # === Quality Gate (Early Rejection) ===
                    # Strategy: "Kill Low Quality"
                    # If the constructed solution is too far from the best known,
                    # we assume it's in a bad basin of attraction and abort immediately.
                    
                    # Dynamic thresholds based on log analysis (20th percentile of good runs)
                    case_name = data.split('.')[0]
                    
                    quality_ratio = env.key_value / env.best_known

                    if env.key_value > current_best:
                        current_best = env.key_value
                        
                        # === Cooperative Search: Share Best Solution ===
                        if self.high_quality_solution_dir:
                            try:
                                # Check if we should dump (is it better than or equal to the pool's best?)
                                # We use >= to allow diversity (multiple runs reaching the same best score)
                                pool_best = self._get_pool_best_value()
                                if current_best >= pool_best:
                                    # Filename format: current_best.{cut_value}.{exp_id}.{run_id}
                                 fname = f"current_best.{int(env.key_value)}.{experiment}.{run_id}"
                                 path = os.path.join(self.high_quality_solution_dir, fname)
                                 env.dump_best_solution(path)
                            except Exception as e:
                                print(f"Failed to dump best solution to pool: {e}")

                        selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                        end = datetime.now()
                        time_cost = (end - begin).total_seconds()
                        init_value = env.key_value
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tConstruction completed")
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tInit:{init_value}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                        if quality_ratio < quality_threshold:
                            print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\t [Quality Gate] Initial score {env.key_value} ({quality_ratio:.1%}) < {quality_threshold:.0%}. Aborting run to restart.", flush=True)
                            return False # Return False to signal the runner to stop this episode
                        else:
                            print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\t [Quality Gate] Initial score {env.key_value} ({quality_ratio:.1%}) >= {quality_threshold:.0%}. Starting improvement phase.", flush=True)
                
            # Phase 2: Improvement (and Perturbation)
            else:
                # POLISHING PHASE: If we are close to best known and stagnating, try all heuristics
                # This is the "Last Mile" optimization.
                # Trigger earlier (50% of stagnation) to catch local optima before ruin
                if not polishing_attempted and no_improve_steps > max_no_improve * 0.5 and current_best >= env.best_known * 0.99:
                     print(f"Run:{run_id} Close to optimum ({current_best}/{env.best_known}). Triggering Polishing Phase.", flush=True)
                     # Try all improvement heuristics once (VND style)
                     for h in self.improvement_heuristics:
                         env.run_heuristic(h)
                         if env.key_value > last_value:
                             print(f"Run:{run_id} Polishing successful with {h.__name__}!", flush=True)
                             last_value = env.key_value
                             no_improve_steps = 0
                             polishing_attempted = False # Reset to allow future polishing
                             break # Go back to main loop to update best etc.
                     
                     if no_improve_steps > 0:
                         polishing_attempted = True # Mark as done for this stagnation cycle
                     continue

                # Check if we need perturbation
                if no_improve_steps > max_no_improve:
                    perturbation_count += 1
                    
                    # Check for Massive Ruin (Continuous Deletion)
                    if perturbation_count > max_perturbations_before_ruin:
                        
                        # === FAIL FAST STRATEGY (Dynamic Restart) ===
                        # Calculate gap to best known
                        gap = 1.0
                        if env.best_known > 0:
                            gap = (env.best_known - current_best) / env.best_known
                        
                        # Threshold for "Close Enough to Dig Deep"
                        # If we are more than the threshold away, we are likely in a bad basin.
                        # Instead of spending hours trying to fix it with Massive Ruin, 
                        # we just FAIL FAST and let the worker pick a new seed.
                        
                        if gap > self.fail_fast_threshold:
                            print(f"Run:{run_id} Stagnated at {current_best} (Gap: {gap:.2%}). Threshold {self.fail_fast_threshold:.2%}. FAIL FAST triggered -> Next Task.", flush=True)
                            return False

                        # Adaptive Logic: Did we improve since the last ruin?
                        if current_best > best_at_last_ruin:
                            # Yes, we improved! Reset ruin intensity.
                            print(f"Run:{run_id} Progress made ({best_at_last_ruin} -> {current_best}). Resetting ruin intensity.", flush=True)
                            current_ruin_percent = 0.3
                            best_at_last_ruin = current_best
                        else:
                            # No, we are stuck in the same basin. Increase intensity.
                            old_ruin = current_ruin_percent
                            current_ruin_percent = min(0.5, current_ruin_percent + 0.05)
                            print(f"Run:{run_id} No progress since last ruin. Intensifying ruin: {old_ruin:.2f} -> {current_ruin_percent:.2f}", flush=True)

                        # EARLY STOPPING: If we are at 50% ruin and still stuck, abandon this run.
                        # The worker will pick up a new run (new seed) from the queue.
                        # Strategy: Fail Fast & Restart with new seed/hybridization
                        if current_ruin_percent >= 0.55:
                            # If we have reached the best known solution, we should not give up.
                            # Instead, we reset the ruin intensity to continue searching (Extended Mode).
                            if current_best >= env.best_known:
                                print(f"Run:{run_id} Reached Best Known ({current_best}). Extending search resources (Resetting Ruin).", flush=True)
                                current_ruin_percent = 0.3
                                best_at_last_ruin = current_best
                            else:
                                print(f"Run:{run_id} STUCK at {current_best} despite max ruin. EARLY STOPPING to change seed.", flush=True)
                                break

                        print(f"Run:{run_id} Stagnated after {perturbation_count} perturbations. MASSIVE RUIN (Backtracking) with {current_ruin_percent:.0%}.", flush=True)
                        
                        # Determine how many nodes to remove
                        nodes_to_remove = max(10, int(node_num * current_ruin_percent))
                        
                        # Select Ruin Strategy
                        # 1. Random Batch Ruin (Default, good for general escape)
                        # 2. Worst Contribution Ruin (Greedy, good for fixing bad decisions)
                        # 3. Cluster Ruin (Spatial, good for escaping local optima traps)
                        
                        ruin_strategy = "random"
                        rand_val = random.random()
                        if rand_val < 0.4:
                            ruin_strategy = "worst"
                        elif rand_val < 0.7:
                            ruin_strategy = "cluster"
                        
                        batch_heuristic = None
                        if ruin_strategy == "worst":
                            batch_heuristic = next((h for h in self.ruin_heuristics if h.__name__ == "batch_worst_ruin"), None)
                        elif ruin_strategy == "cluster":
                            batch_heuristic = next((h for h in self.ruin_heuristics if h.__name__ == "batch_cluster_ruin"), None)
                        
                        # Fallback to random batch ruin
                        if not batch_heuristic:
                             batch_heuristic = next((h for h in self.ruin_heuristics if h.__name__ == "batch_ruin"), None)
                        
                        if batch_heuristic:
                             print(f"  -> Executing Massive Ruin using '{batch_heuristic.__name__}' (Strategy: {ruin_strategy})", flush=True)
                             env.run_heuristic(batch_heuristic, parameters={"count": nodes_to_remove})
                             current_steps += 1
                             removed_count = nodes_to_remove
                        else:
                            removed_count = 0
                            # Continuous deletion loop
                            for _ in range(nodes_to_remove * 2): # Safety factor 2x attempts
                                if removed_count >= nodes_to_remove:
                                    break
                                
                                # FIX: Only use RUIN heuristics (DeleteOperator) for massive ruin
                                # Previously, mutation heuristics (SwapOperator) were mixed in, causing "fake ruin"
                                if self.ruin_heuristics:
                                    heuristic = random.choice(self.ruin_heuristics)
                                    # Avoid batch ruin in loop if it exists in the list but we are here for some reason
                                    if "batch" in heuristic.__name__:
                                         continue
                                         
                                    env.run_heuristic(heuristic)
                                    removed_count += 1
                                    current_steps += 1
                                else:
                                    print("Error: No ruin heuristics available for massive ruin!", flush=True)
                                    break
                        
                        print(f"  -> Removed {removed_count} nodes. Rebuilding (Randomized Mode)...", flush=True)
                        perturbation_count = 0
                        no_improve_steps = 0
                        last_value = env.key_value 
                        continue

                    # Normal (Small) Perturbation
                    # Delete a few nodes (5-20) instead of just 1 to shake it up more
                    # For large graphs, we need stronger perturbation
                    # Adaptive Perturbation: Increase size if we keep stagnating (perturbation_count)
                    base_perturb = max(20, int(node_num * 0.005)) # 0.5% of nodes (e.g. 100 for G81)
                    
                    # Scale up with repeated failures
                    multiplier = 1.0 + (perturbation_count * 0.5)
                    base_perturb = int(base_perturb * multiplier)

                    # REMOVED: High quality protection logic. 
                    # We need strong perturbation to escape local optima, even if we are close to best known.
                    # if current_best > env.best_known * 0.9 and perturbation_count == 0:
                    #    base_perturb = max(10, int(node_num * 0.001)) 
                        
                    perturb_size = random.randint(base_perturb, base_perturb * 2)
                    # === FAIL FAST STRATEGY (Dynamic Restart) ===
                    # Calculate gap to best known
                    gap = 1.0
                    if env.best_known > 0:
                        gap = (env.best_known - current_best) / env.best_known
                    
                    # Threshold for "Close Enough to Dig Deep"
                    # If we are more than the threshold away, we are likely in a bad basin.
                    # Instead of spending hours trying to fix it with Massive Ruin, 
                    # we just FAIL FAST and let the worker pick a new seed.
                    
                    # FIX: Do NOT Fail Fast on small perturbations. Only on Massive Ruin.
                    # Small perturbation is part of the local search process.
                    # if gap > self.fail_fast_threshold:
                    #    print(f"Run:{run_id} Stagnated at {current_best} (Gap: {gap:.2%}). Threshold {self.fail_fast_threshold:.2%}. FAIL FAST triggered -> Next Task.", flush=True)
                    #    return False        
                    # else:            
                    print(f"Run:{run_id} Stagnation ({no_improve_steps} steps). Triggering Small Perturbation (Size: {perturb_size}). Gap: {gap:.2%}", flush=True)

                    for _ in range(perturb_size):
                        # FIX: Do NOT use mutation_heuristics (like Simulated Annealing) in a loop!
                        # SA is a process, not an atomic operator. Running it 200 times is extremely slow.
                        # Only use atomic Ruin (Delete) or Perturbation (Flip) operators here.
                        
                        if self.ruin_heuristics and random.random() < 0.7:
                             heuristic = random.choice(self.ruin_heuristics)
                             # Avoid batch ruin in loop
                             while "batch" in heuristic.__name__ and len(self.ruin_heuristics) > 1:
                                 heuristic = random.choice(self.ruin_heuristics)
                        else:
                             # Fallback to perturbation (random flip)
                             # Filter out SA from perturbation_heuristics if present
                             valid_perturb = [h for h in self.perturbation_heuristics if "simulated_annealing" not in h.__name__]
                             if valid_perturb:
                                 heuristic = random.choice(valid_perturb)
                             elif self.ruin_heuristics:
                                 heuristic = random.choice(self.ruin_heuristics)
                             else:
                                 break # Nothing to do
                             
                        env.run_heuristic(heuristic)
                    
                    no_improve_steps = 0 # Reset counter
                    last_value = env.key_value
                    # After perturbation, we might be incomplete, so next loop will go to Phase 1
                    continue

                # Normal Improvement
                if not active_improvement_heuristics:
                     # If no improvement heuristics, just stop or continue random construction (unlikely)
                     break
                
                # STRATEGIC TABU INJECTION
                # If we are stagnating but not yet ready for perturbation, try Tabu to break free.
                # Trigger frequently to escape local optima
                if hasattr(self, 'tabu_heuristic') and self.tabu_heuristic:
                    # Dynamic frequency based on graph size
                    # Small graph (< 2000): Aggressive Tabu (every 100 steps)
                    # Large graph (> 5000): Conservative Tabu (every 500 steps)
                    tabu_interval = 500
                    if node_num < 2000:
                        tabu_interval = 100
                    
                    # FIX: Ensure Tabu triggers BEFORE max_no_improve (which is 300 for large graphs)
                    if node_num > 5000:
                        tabu_interval = 100

                    # Inject Tabu every 'tabu_interval' steps of stagnation
                    if no_improve_steps > 0 and no_improve_steps % tabu_interval == 0:
                        print(f"Run:{run_id} Stagnation detected ({no_improve_steps}/{max_no_improve}). Injecting Tabu Search.", flush=True)
                        env.run_heuristic(self.tabu_heuristic)
                        current_steps += 1
                        # Check if Tabu helped
                        if env.key_value > last_value:
                            last_value = env.key_value
                            no_improve_steps = 0
                            if env.is_valid_solution and env.key_value > current_best:
                                current_best = env.key_value
                                # Log update...
                        else:
                            no_improve_steps += 1
                        continue # Skip normal improvement this step

                # === UCB Selection Strategy ===
                selected_heuristic = None
                
                # 1. Try untried heuristics first (Cold Start)
                untried = [h for h in active_improvement_heuristics if heuristic_stats[h.__name__]['count'] == 0]
                if untried:
                    selected_heuristic = random.choice(untried)
                else:
                    # 2. Calculate UCB values
                    best_ucb = -float('inf')
                    for h in active_improvement_heuristics:
                        stats = heuristic_stats[h.__name__]
                        avg_reward = stats['reward'] / stats['count']
                        # UCB = Average Reward + Exploration Term
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
                
                # Heartbeat logging for debugging speed
                if current_steps % 100 == 0:
                     print(f"Run:{run_id} Step:{current_steps} NoImprove:{no_improve_steps} Val:{env.key_value} Best:{current_best}", flush=True)
                
                # Check improvement and Update UCB
                improvement = max(0, env.key_value - last_value)
                h_name = selected_heuristic.__name__
                heuristic_stats[h_name]['count'] += 1
                heuristic_stats[h_name]['reward'] += improvement

                if env.key_value > last_value:
                    last_value = env.key_value
                    no_improve_steps = 0
                    polishing_attempted = False # Reset polishing state on any improvement
                    perturbation_count = 0 # Reset perturbation escalation on any improvement
                    if env.is_valid_solution and env.key_value > current_best:
                        current_best = env.key_value
                        selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                        end = datetime.now()
                        time_cost = (end - begin).total_seconds()
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tInit:{init_value}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                        
                        # === High Quality Solution Pool Logic ===
                        # Optimization: Check cache first to avoid unnecessary I/O
                        # Only if we exceed the CACHED pool best do we check the real disk (or just write)
                        # Actually, we can just trust the cache for 60s. If we are better than cache, we try to write.
                        # Writing is safe because dump_best_solution handles atomic writes.
                        pool_best = self._get_pool_best_value()
                        
                        # Throttle writes: Don't write if we just wrote recently (e.g. < 30s) unless it's a massive jump
                        current_time = time.time()
                        last_write_time = getattr(self, '_last_pool_write_time', 0)
                        write_interval = 30 # seconds
                        
                        # Condition: High quality enough (e.g. > 99.5% of pool best) to maintain diversity
                        # We don't want to only save the absolute best, but a population of good seeds.
                        # STRATEGY:
                        # 1. New Global Best: Always save immediately.
                        # 2. Diversity Solution (>= 99.5%): Save probabilistically to avoid flooding.
                        #    - Probability increases as score gets closer to best.
                        #    - Base probability 5% for 99.5% score, up to 100% for best.
                        
                        should_save = False
                        if pool_best == 0:
                            should_save = True
                        elif env.key_value >= pool_best * 0.99:
                            should_save = True
                        elif env.key_value >= pool_best * 0.995:
                            # Probabilistic acceptance for sub-optimal solutions
                            # Linear interpolation: 
                            # Score = 0.995 * Best -> Prob = 0.05
                            # Score = 1.000 * Best -> Prob = 1.00
                            ratio = env.key_value / pool_best
                            acceptance_prob = 0.05 + (ratio - 0.995) / (1.0 - 0.995) * 0.95
                            if random.random() < acceptance_prob:
                                should_save = True
                        
                        if should_save and current_steps > 1:
                             if (current_time - last_write_time > write_interval) or (env.key_value > pool_best):
                                 fname = f"current_best.{int(env.key_value)}.{experiment}.{run_id}"
                                 path = os.path.join(self.high_quality_solution_dir, fname)
                                 env.dump_best_solution(path)
                                 print(f"Run:{run_id} Saved new pool best: {env.key_value} to {fname}", flush=True)
                                 self._last_pool_write_time = current_time
                                 # Update local cache immediately to prevent self-spamming
                                 # We update the set_a as well so we don't save the same solution again immediately
                                 self._pool_best_cache = max(getattr(self, '_pool_best_cache', 0), env.key_value)
                                 self._pool_best_time = current_time
                else:
                    no_improve_steps += 1

                # Logging
                current_best = max(current_best, env.key_value)

                if env.key_value == env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        env.dump_result(result_file=f"match_best_known_result.{experiment}.{run_id}.txt")

                # Check best known
                if env.key_value > env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!", flush=True)
                        print(env.current_solution, flush=True)
                        env.dump_result(result_file=f"break_best_known_result.{experiment}.{run_id}.txt")
                        found_best = True
                        # Don't stop, try to improve more!
                        env.best_known = env.key_value # Update local best known to keep pushing

        return found_best
