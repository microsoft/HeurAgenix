import os
import random
import pickle
import glob
import time
import hashlib
from src.problems.max_cut.env import Env
from src.problems.max_cut.components import Solution
from src.util.util import load_function

class PhasedSearchContinuousHyperHeuristic:
    def __init__(self, heuristic_pool, problem, shared_pool_dir=None, worker_id=None, logger=None, max_restarts=None, **kwargs):
        self.heuristic_pool_names = heuristic_pool
        self.logger = logger
        self.problem = problem
        self.worker_id = str(worker_id)
        self.shared_pool_dir = shared_pool_dir
        self.max_restarts = max_restarts
        self.restart_count = 0
        
        # Initialize heuristic lists
        self.constructive_heuristics = []
        self.improvement_heuristics = []
        self.ruin_heuristics = []
        self.breakout_heuristics = {}  # Keep the dictionary for breakout mapping
        
        # Load and Classify Heuristics
        self._classify_heuristics()
        
        self.elite_pool = []
        self.stagnation_level = 0 
        self.consecutive_massive_ruins = 0 
        
        # Distributed Cooperation Setup
        # worker_id is now set in __init__
            
        # Hash worker_id to get a shard index (0-9)
        self.shard_id = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16) % 10
        
        # Throttling
        self.last_upload_time = 0
        self.last_upload_value = 0
        self.last_sync_time = 0
        self.last_restart_step = -1000 
        
        # [NEW 2026-03-01] Epoch-based Elite Pool Infrastructure
        self.pool_id = 0
        self.pool_type = 'inherit'
        self.pending_rebuild = False # Flag to signal main loop to perform hard reset
        
        # [PHASE 2 CONFIG]
        # Capacity limit for pool expansion. 
        self.POOL_CAPACITY = 10000
        
        # [NEW] Elite Quality Filter Size (N)
        # We only use the top N elites for active optimization (relinking, etc.)
        self.ELITE_FILTER_SIZE = 1000
        
        # [NEW] Migration Count
        # Number of elites to migrate to new pool during inherit expansion
        self.MIGRATION_COUNT = 100

        # [NEW] Diversity Control
        # Minimum distance ratio for diversity checks (0.05 = 5% of nodes)
        self.MIN_DIST_RATIO_MIGRATION = 0.05 
        self.MIN_DIST_RATIO_RELINKING = 0.025 # 2.5% for active relinking warning threshold
        
        if self.shared_pool_dir:
            try:
                self.logger(f"Shared Elite Pool Directory: {self.shared_pool_dir}")
                os.makedirs(self.shared_pool_dir, exist_ok=True)
                
                # Check for existing epoch pools and sync state
                latest_id, latest_type = self._scan_pool_epochs()
                if latest_id > self.pool_id:
                     self.pool_id = latest_id
                     self.pool_type = latest_type
                     self.logger(f"Initialized Pool Pointer to Epoch {self.pool_id} ({self.pool_type})")
                
                # [NEW] Explicitly Create/Log Pool 0 if we are starting fresh
                # If _scan_pool_epochs returns default (0, 'inherit') AND the directory doesn't exist yet, we create it.
                if self.pool_id == 0 and self.pool_type == 'inherit':
                     pool0_path = self._get_pool_path(0, 'inherit')
                     if not os.path.exists(pool0_path):
                         try:
                             os.makedirs(pool0_path, exist_ok=False)
                             self.logger("\n" + "-" * 60 + "\n" + f"  INITIALIZATION: Created First Pool pool_0_inherit" + "\n" + "-" * 60)
                         except FileExistsError:
                             pass # Someone else created it just now
                
                # If no pool exists, self.pool_id / self.pool_type remain default (0, 'inherit')
                # but we need to ensure the directory exists for writing (lazy create in write)
                
            except OSError:
                pass 
                
    def _scan_pool_epochs(self):
        """Scans the shared directory for pool_{id}_{type} folders and updates local pointer to the latest epoch."""
        if not self.shared_pool_dir: return
        
        try:
            entries = os.listdir(self.shared_pool_dir)
            pools = []
            
            for entry in entries:
                # Format: pool_{id}_{type}
                parts = entry.split('_')
                if len(parts) >= 3 and parts[0] == 'pool' and parts[1].isdigit():
                    pid = int(parts[1])
                    ptype = parts[2]
                    # Full path
                    path = os.path.join(self.shared_pool_dir, entry)
                    if os.path.isdir(path):
                        pools.append((pid, ptype))
            
            if pools:
                # Sort by ID ascending
                pools.sort(key=lambda x: x[0])
                latest_id, latest_type = pools[-1]
                
                # [FIX] Do NOT update state here. Just return what we found.
                # State update should be explicit in _check_and_update_pool_id()
                return (latest_id, latest_type)
            
            return (self.pool_id, self.pool_type)
                     
        except Exception as e:
            self.logger(f"Error scanning pool epochs: {e}")
            return (self.pool_id, self.pool_type)

    def _get_pool_path(self, pool_id, pool_type):
        """Returns the directory path for a specific pool epoch."""
        return os.path.join(self.shared_pool_dir, f"pool_{pool_id}_{pool_type}")

    def _check_and_update_pool_id(self):
        """Scans periodically and updates the pool pointer if a new epoch is found (Follower Logic)."""
        old_id = self.pool_id
        new_id, new_type = self._scan_pool_epochs()
        if new_id > old_id:
            self.pool_id = new_id
            self.pool_type = new_type
            
            # [CRITICAL DATA HYGIENE 2026-03-01]
            # If we switch to a 'rebuild' pool (L5 Hard Restart), we MUST clear our local elite pool.
            # Otherwise, the subsequent 'Keep-Alive' logic in _sync_shared_pool (which runs right after this)
            # will upload our OLD dirty elites into the pristine NEW pool, contaminating it instantly.
            if new_type == 'rebuild':
                # Follower's response to Revolution
                self.logger(f"-> [FOLLOW] Detected REVOLUTION (Pool {new_id}_rebuild). Resetting...")
                
                self.pending_rebuild = True
                self.elite_pool = [] # CLEAR IMMEDIATELY
                self.visited_peaks = {}
            else:
                # Follower's response to Expansion
                self.logger(f"-> [EXPAND] Switched to new pool: {new_id} ({new_type}) (Inherit)")

    def _try_trigger_rebuild(self):
        """Attempts to trigger a Global Hard Restart (L5) by creating a 'rebuild' epoch."""
        self.logger("Attempting to trigger L5 Global Hard Restart...")
        
        # 1. Check Current Global State (Race Condition Check)
        latest_id, latest_type = self._scan_pool_epochs()
        
        # Scenario 4 (Corrected): Someone (or even myself effectively) is already in a rebuild epoch.
        # If the latest epoch is 'rebuild', we generally verify if we should join it.
        # Logic: If latest is rebuild, and it's fresh (not full/old), we join/reset instead of creating another one.
        if latest_type == 'rebuild':
             # CASE A: Someone else just created a NEW rebuild pool that I haven't joined yet.
             if latest_id > self.pool_id:
                 self.logger(f"Global Rebuild (pool_{latest_id}) detected. Joining revolution...")
                 self._check_and_update_pool_id()
                 return
             
             # CASE B: I am ALREADY in this rebuild pool (latest_id == self.pool_id).
             # If I am calling this, it means I have stagnation_level=5 WITHIN this rebuild epoch.
             # This implies the current rebuild FAILED to solve the problem.
             # So we MUST trigger a NEW rebuild (pool_{n+1}_rebuild).
             # We continually increment IDs to signify new eras.
             else:
                 self.logger(f"Current Rebuild (pool_{latest_id}) failed (L5 detected). Initiating NEXT Rebuild...")
                 # Do NOT return. Fall through to creation logic below.
                 pass

        # Scenario 2: Inherit happened or Rebuild failed. 
        # Determine the next ID. ALWAYS increment.
        # If latest_id > self.pool_id (someone expanded), we skip that expansion and create rebuild on top.
        # If latest_id == self.pool_id, we just increment.
        
        target_id = latest_id + 1
        target_type = 'rebuild'
        
        new_pool_path = self._get_pool_path(target_id, target_type)
        
        try:
             # Atomic creation
             os.makedirs(new_pool_path, exist_ok=False)
             
             # LOGGING: DISTINCTIVE BLOCK FOR INITIATOR
             self.logger("\n" + "-" * 60 + "\n" + f"REBUILD INITIATED: pool_{target_id}_rebuild (Worker {self.worker_id})" + "\n" + "-" * 60)

             
             # Immediately switch to it
             self.pool_id = target_id
             self.pool_type = target_type
             self.pending_rebuild = True # I triggered it, so I must also reset myself!
             
        except FileExistsError:
             # Scenario 3/4 Race: Someone beat us to creating pool_{target_id}
             # Check what they created
             race_id, race_type = self._scan_pool_epochs()
             
             if race_type == 'rebuild':
                 # Good, they did what we wanted. Join them.
                 self.logger("Rebuild race lost, but goal achieved. Joining...")
                 self._check_and_update_pool_id()
             else:
                 # Scenario 3: They created an INHERIT pool while we wanted REBUILD.
                 # We must NOT settle for inherit. We must try again to create rebuild on top of theirs.
                 self.logger("Conflict: Inherit pool created during Rebuild attempt. Retrying Rebuild on top...")
                 # We simply update to the latest inherited pool.
                 # Since search is still stagnant (we didn't rebuild), the next loop iteration in _run_epoch
                 # will see stagnation_level >= 5 again, and call _try_trigger_rebuild AGAIN.
                 # This time, we will try to build on top of the new inherited pool.
                 self._check_and_update_pool_id()
        
        except Exception as e:
             self.logger(f"Error triggering rebuild: {e}")

    def _estimate_pool_size(self):
        """Estimates current pool size by counting ALL shards (Accurate)."""
        current_path = self._get_pool_path(self.pool_id, self.pool_type)
        if not os.path.exists(current_path): return 0
        
        # [OPTIMIZATION REMOVED]
        # Previously sampled 3 shards. For 10 shards, verifying all is fast enough (ms).
        # This provides accurate capacity control.
        total_files = 0
        
        for s_idx in range(10):
             s_path = self._get_shard_path(current_path, s_idx)
             if os.path.exists(s_path):
                 try:
                     # Using scandir is faster than listdir/glob for just counting
                     # We use a generator expression to avoid building a list in memory
                     count = sum(1 for _ in os.scandir(s_path) if _.is_file())
                     total_files += count
                 except:
                     pass
        
        return total_files

    def _try_expand_pool(self):
        """Checks capacity and attempts to create the next pool epoch if needed (Leader Logic)."""
        
        # 1. Estimate Size
        size = self._estimate_pool_size()
        
        if size < self.POOL_CAPACITY:
            return # Not full yet
            
        self.logger(f"Pool Capacity Reached ({size} > {self.POOL_CAPACITY}). Checking for expansion...")
        
        # 2. Race Condition Check: Does a newer pool ALREADY exist?
        latest_id, latest_type = self._scan_pool_epochs()
        
        if latest_id > self.pool_id:
            # SOMEONE BEAT US TO IT!
            # Just switch.
            self.pool_id = latest_id
            self.pool_type = latest_type
            self.logger(f"-> Switched to new pool: {latest_id} ({latest_type}) (Expansion Preempted)")
            return

        # 3. Create New Pool (Leader Action)
        # Inherit Strategy: Next ID, type 'inherit'
        next_id = self.pool_id + 1
        next_type = 'inherit'
        
        new_pool_path = self._get_pool_path(next_id, next_type)
        
        try:
             # Atomic directory creation (mkdir fails if exists)
             os.makedirs(new_pool_path, exist_ok=False)
             self.logger("\n" + "-" * 60 + "\n" + f"  INHERIT EXPANSION: Created pool_{next_id}_{next_type} from pool_{self.pool_id}" + "\n" + "-" * 60)
             
             # 4. Migrate Top Elites (Seed the new pool)
             # [PHASE 3 OPTIMIZATION] Diversity-Aware Migration
             # Instead of just taking the top 100, we select solutions that are:
             # 1. High Score (Top priority)
             # 2. Distinct (Distance check to avoid cloning the same peak)
             
             # [IMPORTANT 2026-03-01]
             # If this is a 'rebuild' epoch (L5 Hard Restart), we do NOT migrate anything.
             # The goal is to start FRESH. Zero legacy.
             # Migration is ONLY for 'inherit' epochs (Pool Expansion).
             
             if next_type == 'inherit' and self.elite_pool:
                 # Sort by value descending
                 sorted_pool = sorted(self.elite_pool, key=lambda x: x["solution"].cut_value, reverse=True)
                 
                 migrated_elites = []
                 seen_values = []
                 
                 # Parameters for diversity check (Direct usage)
                 
                 # Helper for distance
                 def quick_dist(s1, s2):
                     # Simplified distance check:
                     # 1. Check value difference (Fastest)
                     if abs(s1.cut_value - s2.cut_value) > 1e-3:
                         return 999999 # Treat as different
                     
                     # 2. Check Set Intersection (Slow)
                     d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                     d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                     return min(d1, d2)

                 node_num = len(sorted_pool[0]["solution"].set_a) + len(sorted_pool[0]["solution"].set_b)
                 min_dist = max(10, int(node_num * self.MIN_DIST_RATIO_MIGRATION))
                 
                 for wrapper in sorted_pool:
                     if len(migrated_elites) >= self.MIGRATION_COUNT:
                         break
                         
                     candidate = wrapper["solution"]
                     is_distinct = True
                     
                     # Compare with already selected elites
                     # Limit check to top 20 to speed up (O(N*M))
                     for selected in migrated_elites[:20]: 
                         dist = quick_dist(candidate, selected["solution"])
                         if dist < min_dist:
                             is_distinct = False
                             break
                     
                     if is_distinct:
                         migrated_elites.append(wrapper)
                 
                 # If we filtered too aggressively and have very few, relax and fill up
                 if len(migrated_elites) < 20 and len(sorted_pool) > 20:
                      remaining = [w for w in sorted_pool if w not in migrated_elites]
                      migrated_elites.extend(remaining[:(20 - len(migrated_elites))])
                 
                 # Save to NEW pool
                 # Actually, update ID first, then write.
                 self.pool_id = next_id
                 self.pool_type = next_type
                 
                 count = 0 
                 for wrapper in migrated_elites:
                      self._save_to_shared_pool(wrapper, is_keep_alive=True)
                      count += 1
                 
                 self.logger(f"Migrated {count} DIVERSE elites (from {len(sorted_pool)}) to pool_{next_id}_{next_type}")
             else:
                 # Rebuild or Empty Pool: Just Set ID
                 self.pool_id = next_id
                 self.pool_type = next_type
                 self.logger(f"Initialized Empty Pool: pool_{next_id}_{next_type}") 

                 
        except FileExistsError:
             # Race Condition: Another worker created it milliseconds ago.
             self.logger("Pool creation raced. Switching to winner.")
             self._check_and_update_pool_id()
             
        except Exception as e:
             self.logger(f"Error creating pool: {e}")

    def _classify_heuristics(self):
        # Explicit classifications
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
            "continuous_mean_field_batch", 
            "balanced_random_batch", 
            "weighted_degree_batch", 
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
        
        # Required Breakout Heuristics
        # We look for these specifically to populate self.breakout_heuristics
        breakout_map = {
            "batch_cluster_ruin": ["batch_cluster_ruin"],
            "batch_worst_ruin": ["batch_worst_ruin"],
            "path_relinking": ["path_relinking_guided_perturbation", "path_relinking"],
            "anti_consensus": ["anti_consensus_perturbation"],
            "batch_flip": ["batch_flip_perturbation"]
        }

        # 1. Classify standard pool
        for h_name in self.heuristic_pool_names:
            base_name = os.path.basename(h_name).replace(".py", "")
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in improvement_names:
                self.improvement_heuristics.append(func)
            
            # Map to breakout dictionary
            for key, variations in breakout_map.items():
                if base_name in variations:
                    self.breakout_heuristics[key] = func

        # 2. Fallback: If breakout heuristics were missed in the main pool, try to load them from standard paths
        # This ensures backward compatibility with the original explicit loading logic.
        base_path = "src/problems/max_cut/heuristics"
        ruin_path = "evolved_heuristics.part3"
        
        fallback_map = {
            "batch_cluster_ruin": os.path.join(base_path, ruin_path, "batch_cluster_ruin.py"),
            "batch_worst_ruin": os.path.join(base_path, ruin_path, "batch_worst_ruin.py"),
            "path_relinking": os.path.join(base_path, "path_relinking_guided_perturbation.py"),
            "anti_consensus": os.path.join(base_path, ruin_path, "anti_consensus_perturbation.py"),
            "batch_flip": os.path.join(base_path, ruin_path, "batch_flip_perturbation.py")
        }
        
        for key, path in fallback_map.items():
            if key not in self.breakout_heuristics:
                try:
                    # Check relative or absolute
                    if os.path.exists(path):
                        self.breakout_heuristics[key] = load_function(path, problem=self.problem)
                    else:
                        abs_path = os.path.join(os.getcwd(), path)
                        if os.path.exists(abs_path):
                            self.breakout_heuristics[key] = load_function(abs_path, problem=self.problem)
                except Exception:
                    pass




    def _get_shard_path(self, bucket_path, shard_index):
        return os.path.join(bucket_path, f"shard_{shard_index}")

    def _save_to_shared_pool(self, item, is_keep_alive=False):
        if not self.shared_pool_dir: return
        
        # [CRITICAL HYGIENE FIX] 
        # If we are pending a rebuild, we are essentially a "zombie" holding a solution 
        # from Old Epoch. DO NOT save it to the New Epoch directory.
        if getattr(self, "pending_rebuild", False):
            return
        
        # [RESEARCH] Direct access, item is always a Wrapper Dict
        solution = item["solution"]
        save_obj = item 
            
        current_time = time.time()
        
        # Throttling Logic (Skip if simply frequent updates of same quality, unless keep-alive)
        if not is_keep_alive:
            # [2026-03-01] Relaxed Throttling for Diversity
            # We want to allow saving up to 3 different solutions with same score.
            # So we only throttle if we are bombarding the server with the SAME value extremely fast (e.g. < 300s)
            # giving a chance for the disk check below to filter duplicates.
            if solution.cut_value == self.last_upload_value and (current_time - self.last_upload_time) < 300:
                 return 
            
        try:
           
            # [NEW 2026-03-01] Use Epoch-based Path
            # bucket_path = self._get_time_bucket_path(current_time)
            bucket_path = self._get_pool_path(self.pool_id, self.pool_type)
            
            shard_path = self._get_shard_path(bucket_path, self.shard_id)
            
            # Ensure directories exist (lazy creation)
            if not os.path.exists(shard_path):
                try:
                    os.makedirs(shard_path, exist_ok=True)
                except OSError:
                    pass
            
            # [CRITICAL FIX 2026-02-27] Anti-Homogenization Check (Disk Level)
            # Before writing, check if we already have enough copies of this solution score in the shard.
            # This prevents "Keep-Alive" from flooding the pool with identical solutions (e.g. 5326663.8...).
            # We match the integer part to be fast, then check strict float tolerance relative to filenames if needed.
            # But simply limiting per-integer-bucket is a good heuristic for these large floats.
            
            # Pattern: sol_{val}_...
            # We use a glob to count existing files for this roughly similar score
            score_prefix = f"sol_{int(solution.cut_value)}"
            existing_files = glob.glob(os.path.join(shard_path, f"{score_prefix}*.pkl"))
            
            if len(existing_files) >= 3:
                # Check more strictly: are they actually the same score?
                same_score_count = 0
                target_val = solution.cut_value
                for ef in existing_files:
                    try:
                        # filename format: sol_5326663.8146..._timestamp...
                        fname = os.path.basename(ef)
                        val_str = fname.split("_")[1]
                        val = float(val_str)
                        if abs(val - target_val) < 1e-3:
                            same_score_count += 1
                    except:
                        pass
                
                if same_score_count >= 3:
                    # Too many copies on disk already. Do not save/flood.
                    return

            timestamp_int = int(current_time)
            # Filename: sol_{value}_{timestamp}_{worker}_{rand}.pkl
            filename = f"sol_{solution.cut_value}_{timestamp_int}_{self.worker_id}_{random.randint(1000,9999)}.pkl"
            filepath = os.path.join(shard_path, filename)
            
            # Atomic write
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(save_obj, f)
            os.rename(temp_path, filepath)
            
            # [LOGGING UPDATE] Print the path of the saved elite/breakthrough
            self.logger(f"Saved elite solution to: {filepath}")
            
            # Update throttle stats
            self.last_upload_time = current_time
            self.last_upload_value = solution.cut_value
            
        except Exception as e:
            # Ignore errors (e.g. disk full, permission) to keep running
            pass

    def _sync_shared_pool(self):
        if not self.shared_pool_dir: return
        
        # [NEW 2026-03-01] Phase 2: Check Pool Status & Read
        
        # 1. Update Pool Pointer (Follower Logic)
        self._check_and_update_pool_id()
        
        # [CRITICAL SAFETY 2026-03-01]
        # If we detected a REBUILD (Hard Restart), we must ABORT syncing immediately.
        # We are about to be reset. Any read/write now is dangerous and pointless.
        # Specifically, we must NOT execute the Keep-Alive write below.
        if self.pending_rebuild:
             return 
        
        # 2. Check for Expansion (Leader Logic: Am I the one to expand?)
        # Use simple random probability to avoid all workers checking simultaneously
        if random.random() < 0.05: # 5% chance per sync
             self._try_expand_pool()
        
        # 3. READ: Scan CURRENT epoch pool
        buckets_to_scan = []
        
        current_bucket = self._get_pool_path(self.pool_id, self.pool_type)
        buckets_to_scan.append(current_bucket)
        
        # [NEW Phase 2] If new pool is young (e.g. few files), we might also want to read from previous pool?
        # For now, stick to simple switch.
        
        files_to_read = []
        
        for bucket in buckets_to_scan:
            if not os.path.exists(bucket): continue

            
            # Randomly pick 2-3 shards to check in this bucket (Statistically sufficient)
            # We assume shards 0-9 exist
            shards = random.sample(range(10), 3) 
            
            for s_idx in shards:
                s_path = self._get_shard_path(bucket, s_idx)
                if os.path.exists(s_path):
                    try:
                        # Glob only one shard, much faster
                        shard_files = glob.glob(os.path.join(s_path, "*.pkl"))
                        # Take random sample if too many in one shard
                        if len(shard_files) > 20: 
                            shard_files = random.sample(shard_files, 20)
                        files_to_read.extend(shard_files)
                    except:
                        pass
        
        # Process files
        for fpath in files_to_read:
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    self._add_to_local_pool(data, share=False)
            except:
                pass
                
        # 2. WRITE (Keep-Alive): Periodically re-upload top elites to current bucket
        # Do this roughly every 5th sync (e.g. every ~250 steps)
        # Using simple random check 20%
        if random.random() < 0.2 and self.elite_pool:
            # Pick top 3 from local pool
            # Sort local pool by value descending
            # [COMPATIBILITY] Handle Wrapper
            sorted_pool = sorted(self.elite_pool, key=lambda s: s["solution"].cut_value, reverse=True)
            top_elites = sorted_pool[:3]
            
            for elite in top_elites:
                self._save_to_shared_pool(elite, is_keep_alive=True)

    def _add_to_local_pool(self, item, share=True):
        # [RESEARCH] Direct access, item is always a Wrapper Dict
        solution_ref = item["solution"]
        trajectory_ref = list(item.get("trajectory", []))
        
        # [DYNAMIC TABU STRATEGY 2026-02-19]
        if not hasattr(self, "visited_peaks"):
             self.visited_peaks = {}
             
        TABU_TOLERANCE = 1e-3
        MAX_VISITS_PER_PEAK = 20
        
        for peak_val, count in self.visited_peaks.items():
            if count >= MAX_VISITS_PER_PEAK and abs(solution_ref.cut_value - peak_val) < TABU_TOLERANCE:
                return

        # Add copy of solution to pool
        
        # Deep copy the sets for storage
        new_sol = Solution(set(solution_ref.set_a), set(solution_ref.set_b), solution_ref.cut_value)
        # Create new wrapper
        new_wrapper = {"solution": new_sol, "trajectory": trajectory_ref}
        
        # ------------------------------------------------------------------
        # Strategy 1: Score-based Duplication Check (New Logic)
        # ------------------------------------------------------------------
        # [MODIFIED 2026-02-26] Allow up to 3 solutions with the same score (User Request)
        # However, use a wider tolerance (1e-3) to treat "jittered" values as identical.
        # This prevents the pool from filling with 20 copies of 5317.000001, 5317.000002, etc.
        same_score_count = 0
        pool_tolerance = 1e-3
        
        for existing_wrapper in self.elite_pool:
            if abs(new_sol.cut_value - existing_wrapper["solution"].cut_value) < pool_tolerance:
                same_score_count += 1
        
        if same_score_count >= 3:
            # We already have enough (3) representatives of this score range. Reject.
            return

        # Policy B: Distinct Solution (Add/Evict)
        
        # If pool is not full, just add it.
        if len(self.elite_pool) < 20:
            self.elite_pool.append(new_wrapper)
            if share: self._save_to_shared_pool(new_wrapper)
            return

        # Find worse solution (using wrappers)
        
        pool_values = []
        for e in self.elite_pool:
            idx_val = e["solution"].cut_value
            pool_values.append(idx_val)
            
        min_val = min(pool_values)
        
        # Case 1 & 2: Better or Equal to worst
        if new_sol.cut_value >= min_val:
            # Replace the worst one
            # Note: Since we passed the "Score Check" above, we know we aren't flooding.
            
            for i, s in enumerate(self.elite_pool):
                if s["solution"].cut_value == min_val:
                    self.elite_pool[i] = new_wrapper
                    if share: self._save_to_shared_pool(new_wrapper)
                    break
            return

        # Case 3: Worse than worst -> REJECT
        # [MODIFIED 2026-02-26] Removed "Stranger Admission" logic.
        # We value quality first. If it's worse than the worst elite, it's out.
        else:
             return

    # Removed _calculate_consensus_flip (logic moved to anti_consensus_perturbation heuristic)


    def _update_elite_pool_from_env(self, env):
        # [NEW 2026-02-26] Capture solution + trajectory for full reproducibility
        package = env.export_solution_wrapper()
        self._add_to_local_pool(package, share=True)
    
    def _run_improvement_phase(self, env):
        # [VND Implementation with Strict Hill Climbing]
        # Iterate through all available heuristics until no improvement is found.
        # This converts "relaxed" search into "intensive polishing".
        if not self.improvement_heuristics: return False
        
        # Limit max VND iterations to prevent infinite loops (though strict ascent prevents cycling, costs time)
        # CHANGED: Increased from 10 to 10000000. 
        # Since strict ascent is enforced (env.key_value > start_val), infinite loops are impossible
        # unless the score increases indefinitely, which is impossible for MaxCut.
        # Convergence is guaranteed. We want FULL convergence.
        max_vnd_loops = 10000000 
        total_improved = False
        
        # Pre-shuffle heuristics to improve robustness
        # We work on a copy of the list to shuffle it
        heuristics_queue = list(self.improvement_heuristics)
        
        for loop_idx in range(max_vnd_loops):
            improved_in_this_loop = False
            random.shuffle(heuristics_queue)
            
            for heuristic in heuristics_queue:
                # 1. Snapshot State
                backup_wrapper = env.export_solution_wrapper()
                start_val = backup_wrapper["solution"].cut_value
                
                # 2. Run Heuristic (In-Place Modification)
                try:
                    env.run_heuristic(heuristic)
                except Exception as e:
                    # Sparse logging to prevent explosion if heuristic is fundamentally broken
                    if not hasattr(self, "_error_log_count"): self._error_log_count = 0
                    self._error_log_count += 1
                    if self._error_log_count < 10 or self._error_log_count % 1000 == 0:
                         self.logger(f"Error running heuristic {heuristic.__name__}: {e}")
                    
                    env.import_solution_wrapper(backup_wrapper)
                    continue

                # 3. Acceptance Criteria: Strict Ascent
                # If Score Dropped or Equal -> Revert (We want to find peaks, not drift)
                if env.key_value <= start_val:
                    # Revert
                    env.import_solution_wrapper(backup_wrapper)
                    
                    # Note: We assume env.problem_state is derived from current_solution, 
                    # but heuristic might modify algorithm_data too. Usually negligible for basic heuristics.
                else:
                    # Accepted
                    improved_in_this_loop = True
                    total_improved = True
                    # Optimization: If we found a gain, we might want to stick with this heuristic or continue?
                    # Standard VND continues to next heuristic.
            
            # If a full pass through all heuristics yielded no gain, we are at a local optimum for ALL neighborhoods.
            if not improved_in_this_loop:
                break
                
        return total_improved

    def _apply_breakout(self, env, strategy):
        node_num = env.instance_data["node_num"]

        # [NEW] Prepare Filtered Elite Pool for Breakout
        # We only expose the Top N elites to the heuristic engine
        filtered_elites = []
        if self.elite_pool:
             # Sort desc
             sorted_pool = sorted(self.elite_pool, key=lambda s: s["solution"].cut_value, reverse=True)
             # Take top N
             top_n = sorted_pool[:self.ELITE_FILTER_SIZE]
             filtered_elites = [s["solution"] for s in top_n]
        
        if strategy == "supernova_ruin":
            # "Anti-Consensus" Strategy: Flip stable variables
            # Increase intensity to 10%-20% to escape deep basin
            ratio = random.uniform(0.10, 0.20)
            
            # Prepare algorithm context for heuristic (unpack Elite Pool wrappers)
            if filtered_elites:
                env.algorithm_data["elite_pool"] = filtered_elites
            
            # Use evolved heuristic "anti_consensus"
            if "anti_consensus" in self.breakout_heuristics:
                 h = self.breakout_heuristics["anti_consensus"]
                 env.run_heuristic(h, parameters={"ratio": ratio})
                 self.logger(f"Supernova Ruin applied: Anti-Consensus Flip (Ratio: {ratio:.2f}).")
            else:
                 # Fallback if heuristic missing (should be loaded by default)
                 self.logger("Supernova Ruin: Heuristic missing, falling back to Heavy Ruin.")
                 self._apply_breakout(env, "heavy_ruin")

        elif strategy == "active_pool_relinking":
            # [NEW] Active strategy: Force path relinking between distant elites
            if len(self.elite_pool) < 2:
                 return

            # [OPTIMIZED] Use Filtered Pool for Target Selection
            # We want to link with HIGH QUALITY elites (Top N), not just any random elite in the 10000 pool.
            # Using self.ELITE_FILTER_SIZE
            
            candidate_pool = self.elite_pool
            if len(self.elite_pool) > self.ELITE_FILTER_SIZE:
                 # Sort desc and take top N
                 sorted_pool = sorted(self.elite_pool, key=lambda s: s["solution"].cut_value, reverse=True)
                 candidate_pool = sorted_pool[:self.ELITE_FILTER_SIZE]
            
            # 1. Find Best Known (Handle Wrapper) from the FULL pool (usually same as filtered, but just in case)
            best_wrapper = max(self.elite_pool, key=lambda s: s["solution"].cut_value)
            best_sol = best_wrapper["solution"]
            
            # 2. Find a "Distant" High-Quality Elite
            # [FIX] Lower threshold to 0.97 to ensure our current elites (2400-2406) can participate
            # 2446 * 0.97 = 2372, so 2400+ are valid candidates
            # Use candidate_pool instead of self.elite_pool
            candidates = [s for s in candidate_pool if s["solution"].cut_value > env.best_known * 0.97]
            if not candidates: 
                 return
                 
            # Helper for distance
            def calc_dist(s1, s2):
                d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                return min(d1, d2)
            
            # Find candidate farthest from best_sol (Need to unwrap candidate for dist calculation)
            # [FIX] Use weighted random choice to avoid "Groundhog Day" repeating the same link
            # Pick from top 3 farthest
            candidates.sort(key=lambda s: calc_dist(s["solution"], best_sol), reverse=True)
            top_candidates = candidates[:min(3, len(candidates))]
            distant_wrapper = random.choice(top_candidates)
            distant_elite = distant_wrapper["solution"]
            
            dist = calc_dist(best_sol, distant_elite)
            
            # [IMPROVED 2026-02-18] Dynamic Threshold based on Graph Size
            # Hardcoding 'dist < 40' or '200' is bad because it ignores problem scale.
            # Use 1.5% of total nodes as the "too close" threshold.
            
            # CRITICAL FIX: node_num is not in self.problem, it is in env.instance_data
            node_num = env.instance_data["node_num"]
            # [FIX 2026-02-19] Increased safety radius to 2.5% to prevent black hole collapse
            threshold = max(10, int(node_num * self.MIN_DIST_RATIO_RELINKING))

            if dist < threshold: 
                # [FIX 2026-02-19] Improved Robustness:
                # If targets are too close, standard Path Relinking is weak.
                # Instead of skipping or punishing, we force a "Micro-Perturbation" to break strict convergence.
                # This helps exploring the immediate neighborhood of the basin.
                self.logger(f"Active Relinking: Targets too close (Dist={dist} < Threshold={threshold}). Triggering Micro-Perturbation.")
                 
                # Load best solution (WITH TRAJECTORY)
                env.import_solution_wrapper(best_wrapper)
                 
                # Perturb 2% of nodes (enough to move away ~60 nodes in 3000)
                # This is lighter than Level 1 Stagnation (Light Ruin), keeping us in the same "Peak Family".
                h = self.breakout_heuristics["batch_flip"]
                env.run_heuristic(h, parameters={"ratio": 0.02}) 
                return

            self.logger(f"*** ACTIVE RELINKING: Best({best_sol.cut_value}) <-> Distant({distant_elite.cut_value}, Dist={dist}) ***")

            # 3. Reset to Best, Target = Distant (WITH TRAJECTORY)
            env.import_solution_wrapper(best_wrapper)
            
            # Pass unwrapped distant elite
            env.algorithm_data["elite_pool"] = [distant_elite] 
                
            if "path_relinking" in self.breakout_heuristics:
                h = self.breakout_heuristics["path_relinking"]
                # Move 40% towards the other peak
                env.run_heuristic(h, parameters={"intensity": 0.4})
                
                # [FIX]: Immediate Local Optimization in the Valley
                if self.improvement_heuristics:
                     self.logger("Rapid Mining in Valley...")
                     # Execute improvement to settle into a local optimum
                     self._run_improvement_phase(env)

        elif strategy == "path_relinking_to_best" and "path_relinking" in self.breakout_heuristics:
             # Targeted PR: Force link towards the absolute Best Known in the pool
             h = self.breakout_heuristics["path_relinking"]
             # Filter pool to only include the BEST solution(s) to guarantee target direction
             best_val = max(s["solution"].cut_value for s in self.elite_pool)
             # Unwrap solutions for heuristics
             best_solutions = [s["solution"] for s in self.elite_pool if s["solution"].cut_value == best_val]
             
             # Create a focused context
             env.algorithm_data["elite_pool"] = best_solutions
             # Run with high intensity
             env.run_heuristic(h, parameters={"intensity": 1.0})
             # Restore full pool for other operations (though reference is passed, we shouldn't damage self.elite_pool)
             # NOTE: Since we pass list by ref, safest is to NOT modify self.elite_pool. 
             # But here we temporarily overwrote algorithm_data entry, which is fine.
             self.logger(f"Targeted Path Relinking -> Best Known ({best_val})")

             # [FIX] Dig deeper around the path
             if self.improvement_heuristics:
                 self.logger("Mining Path to Best...")
                 self._run_improvement_phase(env)

        elif strategy == "path_relinking" and "path_relinking" in self.breakout_heuristics:
            # Parameters: intensity
            h = self.breakout_heuristics["path_relinking"]
            # Pass elite_pool via algorithm_data
            # We need to hack/inject elite_pool into algorithm_data if not present
            # env.run_heuristic passes env.problem_state and env.algorithm_data
            env.algorithm_data["elite_pool"] = [s["solution"] for s in self.elite_pool]
            env.run_heuristic(h, parameters={"intensity": 0.3})
            
        elif strategy == "jump_to_secondary_peak":
             # Strategy: Teleport to a high-quality local optimum
             if not self.elite_pool:
                 self._apply_breakout(env, "supernova_ruin")
                 return
             
             best_val = max(s["solution"].cut_value for s in self.elite_pool)
             # Candidates: High quality but strictly less than Best Known (to find secondary peaks)
             # We want to revisit peaks like 26992 to see if we can sharpen them
             # [FIX] Use relative threshold for large-weight instances (imgseg)
             threshold = max(150, env.best_known * 0.02)
             
             # Filter wrappers/items based on unwrapped value
             candidates = [s for s in self.elite_pool if s["solution"].cut_value >= env.best_known - threshold and s["solution"].cut_value < best_val]
             
             desc = "SECONDARY PEAK"

             # [FIX] If no secondary peak, jump to a DISTANT parallel peak (same best value)
             if not candidates:
                 def calc_dist_j(s1, s2):
                    d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                    d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                    return min(d1, d2)
                 
                 # Look for solutions with SAME best value but Distance > 400
                 # Handle wrapper
                 candidates = [s for s in self.elite_pool if abs(s["solution"].cut_value - best_val) <= 1e-3 and calc_dist_j(s["solution"], env.current_solution) > 400]
                 desc = "PARALLEL UNIVERSE PEAK"
             
             if candidates:
                 target_item = random.choice(candidates)
                 target_sol = target_item["solution"]
                 
                 # Restore using standard wrapper interface
                 env.import_solution_wrapper(target_item)

                 self.logger(f"*** JUMPED TO {desc}: {env.key_value} (from pool of {len(candidates)}) ***")
             else:
                 # If no secondary peak found, try Supernova
                 self._apply_breakout(env, "supernova_ruin")

        elif strategy == "light_ruin":
            # Cluster Ruin 1%
            if "batch_cluster_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_cluster_ruin"]
                count = max(10, int(node_num * 0.01))
                env.run_heuristic(h, parameters={"count": count})
            else:
                # Fallback
                self._apply_breakout(env, "medium_ruin")
                
        elif strategy == "medium_ruin":
             # Worst Ruin 5%
            if "batch_worst_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_worst_ruin"]
                count = max(50, int(node_num * 0.05))
                env.run_heuristic(h, parameters={"count": count})
        
        elif strategy == "heavy_ruin":
            # Cluster Ruin 10-20%
            if "batch_cluster_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_cluster_ruin"]
                count = max(200, int(node_num * 0.15))
                env.run_heuristic(h, parameters={"count": count})

        elif strategy == "massive_ruin":
            # [OPTIMIZED 2026-02-16]
            # Strategic Reconstructive Ruin (20%-50% Destruction + Cosm Repair)
            # This balances Exploration (jumping out of basin) and Exploitation (using learned structure).
            
            if "batch_worst_ruin" in self.breakout_heuristics:
                h_ruin = self.breakout_heuristics["batch_worst_ruin"]
                
                # Dynamic Ratio: 20% to 50%
                # Lower bound (20%) allows "Large Step" optimization
                # Upper bound (50%) allows "Basin Hopping"
                ratio = random.uniform(0.20, 0.50)
                count = int(node_num * ratio)
                
                self.logger(f"RECONSTRUCTIVE RUIN: Removing {count} nodes ({ratio:.1%}) to trigger repair...")
                env.run_heuristic(h_ruin, parameters={"count": count})
                
                # [Repair Phase]
                # Essential: Use Cosm to fill the holes intelligently
                if self.constructive_heuristics:
                    repair_h = [h for h in self.constructive_heuristics if "cosm" in h.__name__ and "detailed" in h.__name__]
                    if not repair_h:
                        repair_h = [h for h in self.constructive_heuristics if "cosm" in h.__name__]
                    
                    if repair_h:
                         # self.logger(f"Repairing with {repair_h[0].__name__}...")
                         # Cosm checks solution state and fills unselected_nodes
                         env.run_heuristic(repair_h[0])
                
                # [MODIFIED 2026-02-18] Add Noise Injection to prevent Loop
                # Even after ruin, COSM might reconstruct the exact same solution.
                # We force a small random perturbation (5%) to ensure we land in a NEW basin.
                ratio_noise = 0.05
                h = self.breakout_heuristics["batch_flip"]
                env.run_heuristic(h, parameters={"ratio": ratio_noise})
                self.logger(f"Noise Injection: Random Flip ({ratio_noise:.1%}) to escape basin.")

            else:
                # Fallback: Random Flip 40%
                h = self.breakout_heuristics["batch_flip"]
                env.run_heuristic(h, parameters={"ratio": 0.40})
                self.logger("Fallback Ruin: Random Flip (40%).")

        elif strategy == "soft_restart":
             self.logger("... Soft Restart Triggered ... Abandoning current solution.")
             
             # Option A: Jump to a random Elite (preferably one we haven't visited lately)
             force_constructive = False
             
             if self.elite_pool and len(self.elite_pool) > 5:
                  # Pick a random elite, but favor those DIFFERENT from current
                  # Calculate distance to current
                  def calc_dist_r(s1, s2):
                        d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                        d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                        return min(d1, d2)
                  
                  # Sort by distance descending (furthest first)
                  sorted_elites = sorted(self.elite_pool, key=lambda s: calc_dist_r(s["solution"], env.current_solution), reverse=True)
                  # Pick from top 5 furthest
                  target_item = random.choice(sorted_elites[:5])
                  target_sol = target_item["solution"]
                  
                  # [CRITICAL FIX 2026-02-19] Check if the "furthest" elite is actually distant.
                  # If the pool has collapsed (Homogenized), the furthest elite might be just 10 flips away.
                  # In that case, a Soft Restart is useless. We must force a HARD RESTART (Constructive).
                  dist = calc_dist_r(target_sol, env.current_solution)
                  min_restart_dist = max(50, int(node_num * 0.05)) # e.g. 150 nodes for 3000 node graph
                  
                  if dist < min_restart_dist:
                       self.logger(f"Soft Restart Aborted: Pool Homogenized (Max Dist={dist} < {min_restart_dist}). Forcing Hard Constructive Restart.")
                       force_constructive = True
                  else:
                       # Restore using standard wrapper interface
                       env.import_solution_wrapper(target_item)

                       self.logger(f"Restarted from Distant Elite (Val: {target_sol.cut_value}, Dist: {dist})")
             
             else:
                  force_constructive = True
                  
             if force_constructive:
                  # Option B: Complete Noise Restart (if pool is empty or small OR homogenized)
                  # Or Constructive Restart
                  self.logger("Restarting with Constructive Heuristic (High Quality)...")
                  env.clear_solution()
                  
                  # [SYNC WITH COLD START] Use best constructive heuristics to reach High Basin
                  construction_steps = 0
                  while not env.is_complete_solution and construction_steps < 1000:
                      if not self.constructive_heuristics:
                          break
                      
                      # Strict Priority: Detailed > Quick > Mean Field
                      if construction_steps == 0:
                           detailed_cosm = [h for h in self.constructive_heuristics if "cosm_heuristic_detailed" in h.__name__]
                           other_cosm = [h for h in self.constructive_heuristics if ("cosm" in h.__name__ or "mean_field" in h.__name__) and "detailed" not in h.__name__]
                           
                           if detailed_cosm:
                               h = detailed_cosm[0] # Always pick detailed if available
                           elif other_cosm:
                               h = random.choice(other_cosm)
                           else:
                               h = random.choice(self.constructive_heuristics)
                      else:
                          h = random.choice(self.constructive_heuristics)
                          
                      env.run_heuristic(h)
                      construction_steps += 1
                  
                  # [FIX 2026-02-19] Set Immunity Timer
                  self.last_restart_step = self.current_run_steps
                  self.logger(f"Immunity Activated for 500 steps (Restart Step: {self.current_run_steps})")


    def run(self, env: Env) -> bool:
        """
        Main entry point. Wraps the actual search epoch in a loop to handle Global Hard Restarts (L5).
        """
        while True:
            # Run one epoch. Outcomes:
            # 1. Complete successfully (Time limit reached) -> Returns True
            # 2. Critical Failure (Construction failed) -> Returns False
            # 3. L5 Rebuild Triggered -> Set self.pending_rebuild = True, Break Loop
            
            result = self._run_epoch(env)
            
            if self.pending_rebuild:
                self.restart_count += 1
                if self.max_restarts is not None and self.restart_count > self.max_restarts:
                    self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}) - ABORTING due to max_restarts={self.max_restarts} limit reached.")
                    return result
                
                self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}) - Restart {self.restart_count}/{self.max_restarts if self.max_restarts else 'inf'}")
                
                # Reset Flags (Except pending_rebuild which must protect the reset phase)
                self.stagnation_level = 0
                self.consecutive_massive_ruins = 0
                if hasattr(self, 'phase_retries'):
                    self.phase_retries = 0
                
                # Reset Environment Logic
                # Reuse output dir
                # Note: env.reset() clears solution and trajectory
                env.reset(output_dir=env.output_dir, temperature_scaling=1000.0)
                
                # Clear local elite pool to match the new epoch (Blank Slate)
                self.elite_pool = []
                self.visited_peaks = {}
                
                # Sync to ensure we are pointing to the correct rebuild pool
                self._sync_shared_pool()
                
                # ONLY NOW that all local state has synced and initialized to the fresh environment,
                # we disengage the safety lock preventing old outputs.
                self.pending_rebuild = False
                
                continue # Restart Outer Loop
            
            return result

    def _run_epoch(self, env: Env) -> bool:
        # [REFACTORED for Cooperative Search - Cold Start Only]
        
        # Explicitly maximize chances by syncing first (populate pool for interactions later)
        self._sync_shared_pool()

        # Outer Loop for Hard Restart (L5)
        # We wrap the entire search process so we can restart from scratch if a Rebuild epoch is triggered
        # while True: <-- REMOVED, Handled in run() wrapper
            
            # Reset flags for new epoch
            # self.pending_rebuild = False <-- REMOVED
            # self.stagnation_level = 0 <-- REMOVED
            
        self.logger("Switching to Constructive Phase (Cold Start)...")
            # Fallback: Construct New Solution if no Best Known file
        # Loop until solution is COMPLETE and VALID
        max_retries = 10
        for retry in range(max_retries):
            
            # Clear solution for a fresh start while preserving algorithm_data
            env.clear_solution()
            
            # Keep constructing until complete
            construction_steps = 0
            while not env.is_complete_solution and construction_steps < 1000:
                if not self.constructive_heuristics:
                    break
                
                # Prefer Cosm for first attempt as it is SOTA for these graphs
                if retry == 0 and construction_steps == 0:
                        # Strict Priority: Detailed > Quick > Mean Field
                        detailed_cosm = [h for h in self.constructive_heuristics if "cosm_heuristic_detailed" in h.__name__]
                        other_cosm = [h for h in self.constructive_heuristics if ("cosm" in h.__name__ or "mean_field" in h.__name__) and "detailed" not in h.__name__]
                        
                        if detailed_cosm:
                            h = detailed_cosm[0] # Always pick detailed if available
                        elif other_cosm:
                            h = random.choice(other_cosm)
                        else:
                            h = random.choice(self.constructive_heuristics)
                else:
                    h = random.choice(self.constructive_heuristics)
                    
                env.run_heuristic(h)
                construction_steps += 1
            
            if env.is_complete_solution and env.key_value > 100:
                self.logger(f"Construction completed. Value: {env.key_value}")
                break
            else:
                self.logger(f"Construction failed or incomplete (Value: {env.key_value}). Retrying ({retry+1}/{max_retries})...")
        
        if not env.is_complete_solution:
                self.logger("Critical Failure: Unable to construct valid solution after retries.")
                return False
            
        current_best = env.key_value
        self._update_elite_pool_from_env(env)
        
        no_improve_steps = 0

        # CHANGE: Use instance variable to track steps for coordination with restart logic
        self.current_run_steps = 0
        
        self.logger(f"Starting Breakout Search from {current_best}...")

        # [DYNAMIC TABU LIST]
        # Keep track of local optima we have visited frequently.
        # Format: {cut_value: visit_count}
        self.visited_peaks = {}
        # [FIX: RELAX TOLERANCE]
        # Strict 1e-6 is too tight for float variations on different heuristic paths.
        # Use 1e-3 (or even 0.1) since object function values are usually large integers/floats.
        # For MaxCut with float weights, peaks are usually separated by significant margins.
        TABU_TOLERANCE = 1e-3 
        MAX_VISITS_PER_PEAK = 20 # How many times can we rediscover the same peak before banning it?

        while env.continue_run:
            self.current_run_steps += 1
            
            # [DYNAMIC ANTI-GRAVITY SHIELD]
            # Instead of hardcoding, we check if the current value has been "exhausted".
            current_val = env.key_value
            is_tabu = False
            
            # Check if we are in a Forbidden Peak
            for peak_val, count in self.visited_peaks.items():
                if count >= MAX_VISITS_PER_PEAK and abs(current_val - peak_val) < TABU_TOLERANCE:
                     is_tabu = True
                     # Only log sparingly
                     if self.current_run_steps % 100 == 0:
                         self.logger(f"In Exhausted Basin ({peak_val}). Triggering Evacuation.")
                     break
            
            if is_tabu:
                # Force massive ruin (Supernova) to escape processing this dead zone
                # [FIX 2026-02-27] Preventing infinite loop if heuristic fails to change solution
                prev_tabu_val = env.key_value
                self._apply_breakout(env, "supernova_ruin")
                
                # Check if we actually moved out of the basin
                if abs(env.key_value - prev_tabu_val) < 1e-3:
                     self.logger("Supernova failed to break Tabu (Value unchanged). Forcing Random Ruin.")
                     # Fallback to pure random ruin which is guaranteed to change state
                     # Use batch_flip explicitly if available, otherwise massive_ruin which has fallbacks
                     if "batch_flip" in self.breakout_heuristics:
                         h = self.breakout_heuristics["batch_flip"]
                         # Flip 20%
                         env.run_heuristic(h, parameters={"ratio": 0.20})
                     else:
                         self._apply_breakout(env, "massive_ruin")
                     
                # [SAFETY BREAK 2026-02-27] Final check to prevent ANY infinite loop
                if abs(env.key_value - prev_tabu_val) < 1e-3:
                     self.logger("CRITICAL: Breakout failed to change solution. Forcing escape from Tabu block.")
                     # If we can't move, we must let the main loop proceed, 
                     # even if it means researching the same peak (which will trigger standard stagnation logic).
                     # We might be at a global optimum where no move is possible? (Unlikely for MaxCut)
                     pass
                else:
                    # Successful move, loop back to start to re-evaluate new position
                    continue

            # --- Phase A: Repair / Improve ---
            # Try to improve current solution (which might be ruined)
            improved = self._run_improvement_phase(env)
            
            # [CRITICAL FIX 2026-02-26] Detect Stagnation at Local Optima
            # If we are at a local optimum (not improving), we must record this peak
            # to prevent infinite cycling around the same basin.
            if not improved:
                 peak_val = env.key_value
                 found_family = False
                 for existing_val in list(self.visited_peaks.keys()):
                     if abs(peak_val - existing_val) < TABU_TOLERANCE:
                         self.visited_peaks[existing_val] += 1
                         found_family = True
                         break
                 if not found_family:
                     self.visited_peaks[peak_val] = 1
            
            # --- Phase B: Check Status ---
            # 1. Update Global/Local Best
            # [2026-02-28] Use threshold to prevent micro-fluctuations (1e-6) from resetting stagnation
            # Only count as improvement if gain > 1e-3
            if env.key_value > current_best + 1e-3:
                current_best = env.key_value
                
                # [DYNAMIC TABU LIST UPDATE]
                # We found a new peak. Record it.
                # Linear scan to see if it belongs to an existing family
                found_family = False
                for existing_val in list(self.visited_peaks.keys()): # List copy as we might modify
                    if abs(current_best - existing_val) < TABU_TOLERANCE:
                        # Update the peak definition to the better value
                        count = self.visited_peaks[existing_val]
                        del self.visited_peaks[existing_val]
                        self.visited_peaks[current_best] = count + 1
                        found_family = True
                        break
                
                if not found_family:
                    # New distinct peak
                    self.visited_peaks[current_best] = 1
                
                no_improve_steps = 0
                self.stagnation_level = 0
                self.consecutive_massive_ruins = 0 # Reset panic counter on improvement
                self._update_elite_pool_from_env(env)
                self.logger(f"Step:{self.current_run_steps} NEW LOCAL BEST: {current_best}")
                

                if current_best > env.best_known:
                    self.logger(f"!!! BREAKTHROUGH: {current_best} > {env.best_known} !!!")
                    env.best_known = current_best
                    
                    # [SAFE SAVE STRATEGY] Only save if strictly better than anything on disk
                    saved_best = 0.0
                    if os.path.exists(env.output_dir):
                        for f in os.listdir(env.output_dir):
                            if f.startswith("breakthrough_") or f.startswith("match_"):
                                try:
                                    # Format: breakthrough_..._SCORE.txt
                                    part = f.rsplit("_", 1)[-1] 
                                    score = float(part.replace(".txt", ""))
                                    if score > saved_best:
                                        saved_best = score
                                except: pass
                    
                    if (current_best - saved_best) > 1e-3:
                        env.dump_result(result_file=f"breakthrough_from_worker_{self.worker_id}_{current_best}.txt")

                elif abs(current_best - env.best_known) < 1e-3:
                     self.logger(f"~~~ MATCHED BEST KNOWN: {current_best} ~~~")
                     # Only save Match if no results exist yet
                     has_records = False
                     if os.path.exists(env.output_dir):
                         for f in os.listdir(env.output_dir):
                             if f.startswith("breakthrough_") or f.startswith("match_"):
                                 has_records = True
                                 break
                     
                     if not has_records:
                        env.dump_result(result_file=f"match_best_known_from_worker_{self.worker_id}_{current_best}.txt")
            else:
                no_improve_steps += 1
                
                # 2. Relaxed Elite Pool Update (Fix for Path Relinking)
                # If we are stuck but the solution is still decent (e.g. > 98% of BK)
                # we add it to the pool to provide diversity for path relinking.
                # Don't add every step, maybe every 10 steps to avoid flooding with identical copies
                is_best_known = env.key_value >= env.best_known
                if is_best_known or (env.key_value >= env.best_known * 0.98 and self.current_run_steps % 10 == 0):
                    self._update_elite_pool_from_env(env)

            # --- Phase C: Breakout / Ruin Strategies ---
            # Adaptive Patience based on problem size
            # Large instances (e.g. 100k nodes) need significant time to explore deep basins.
            node_num = env.instance_data.get("node_num", 1000)
            base_patience = max(500, int(node_num / 10)) # e.g. 100k nodes -> 10,000 steps check interval? No, keep it responsive.
            # Updated 2026-02-24: Use a balanced patience. 
            # For 100k nodes, we check every ~2000 steps.
            patience = min(2000, max(200, int(node_num / 50)))
            
            if no_improve_steps > patience:
                # Instead of immediate level jump, we use a "Retry Budget" based on graph size.
                # Complex graphs need more retries at each intensity level before giving up.
                
                # Dynamic Retry Thresholds calculated from node_num
                # Small graph (800): ~4 retries. Large graph (100k): ~20 retries allowed per phase.
                # Logic: Don't give up strictly on a strategy until we've tried it enough times relative to problem complexity.
                # [2026-02-28] UPDATE: Log analysis (imgseg_103041, 126039, 135037) shows >1 retry per phase is futile.
                # All successful breakouts happen on Attempt 0 or 1. Escalation is better.
                # [Request 2026-02-28] Increase to 2 retries to give each level slightly more chance before nuclear option.
                max_retries_per_phase = 2 
                
                # Increment internal counter for current phase
                if not hasattr(self, 'phase_retries'):
                    self.phase_retries = 0
                
                self.phase_retries += 1
                
                # Map simple 4 levels (1, 2, 3, 4) based on how many retries we've exhausted
                # We stay in strict 4 phases. Escalation happens only when phase_retries exceeds budget.
                
                if self.stagnation_level == 0:
                    self.stagnation_level = 1 # Start stagnation handling
                    self.phase_retries = 1 # [2026-02-28] Start at 1 for clearer logging (Try 1/2, 2/2)
                elif self.phase_retries > max_retries_per_phase:
                     # Budget exhausted for current level, escalate!
                     
                     # [TEST MODE: Fast Forward L1->L4] Reverted to normal logic
                     self.stagnation_level += 1
                     
                     self.phase_retries = 1 # Reset for new level (Start at 1)
                     self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted {max_retries_per_phase} retries)")

                # [OPTIMIZED HIERARCHY 2026-03-01: 5-Level Logic with Race Handling]
                strategy = "heavy_ruin" # Fallback
                
                # Check relation to Elite Pool (Global Best)
                is_attacking_global_best = False
                global_best_val = 0
                if self.elite_pool:
                     global_best_val = max(s["solution"].cut_value for s in self.elite_pool)
                     # Using 1e-3 tolerance
                     if current_best >= global_best_val - 1e-3:
                         is_attacking_global_best = True
                
                if self.stagnation_level >= 5:
                     # L5: Global Hard Restart
                     # [CONSTRAINT 2026-03-01] Only trigger if we are actively attacking the Global Best
                     # and have failed multiple times (implied by reaching Level 5).
                     # If we are just a weak worker failing locally, we shouldn't reset everyone.
                     
                     if is_attacking_global_best:
                         self.logger(f"Stagnation L5. Qual={env.key_value:.0f} Act=hard_restart")
                         self.logger(f"Step:{self.current_run_steps} L5 Detected (Attacking Global Best {global_best_val})! -> ATTEMPTING REVOLUTION")
                         self._try_trigger_rebuild()
                         
                         if self.pending_rebuild:
                             break # Break loop to restart
                         else:
                             strategy = "soft_restart" # Fallback
                     else:
                         self.logger(f"Step:{self.current_run_steps} L5 Detected, but Local Best ({current_best}) < Global Best ({global_best_val}). Downgrading to Soft Restart.")
                         strategy = "soft_restart"
                
                elif self.stagnation_level == 4:
                     # Level 4: Soft Restart (The "Nuclear" Option)
                     # Standard Soft Restart to random distant elite or constructive
                     strategy = "soft_restart"
                
                elif self.stagnation_level == 3:
                     # Level 3: Supernova Ruin (Anti-Consensus)
                     strategy = "supernova_ruin"
                     
                elif self.stagnation_level == 2:
                     # Level 2: Massive Reconstructive Ruin
                     strategy = "massive_ruin"
                     
                elif self.stagnation_level == 1:
                     # Level 1: Diversification
                     if len(self.elite_pool) > 2 and random.random() < 0.6:
                         strategy = "path_relinking_to_best"
                     elif random.random() < 0.5:
                         strategy = "jump_to_secondary_peak"
                     else:
                         strategy = "heavy_ruin" 

                self.logger(f"Step:{self.current_run_steps} Stagnation L{self.stagnation_level} (Try {self.phase_retries}/{max_retries_per_phase}). Qual={env.key_value:.0f} Act={strategy}")
                
                # [CRITICAL LOGIC FIX 2026-03-01] Remove Premature Resets
                # Do NOT reset stagnation_level = 0 here. 
                # Doing so prevents us from climbing the ladder (L1->L2->...->L5).
                # The accumulation of phase_retries will push us to the next Level.
                # Only if the strat SUCCEEDS (finds improvement) do we reset in the improvement check above.
                
                self._apply_breakout(env, strategy)
                
                # [BUG FIX 2026-02-21] Detect Hard Restart and Reset Baseline
                # If breakout resulted in a massive value drop (e.g. > 10%), it means we restarted.
                # We must reset current_best to avoid immediate stagnation detection (comparing against the old peak).
                if env.key_value < current_best * 0.90:
                    self.logger(f"Significant Value Drop ({current_best} -> {env.key_value})")
                    
                    # [CRITICAL FIX 2026-03-01] Handle Baseline Resets Correctly
                    # 1. current_best: 
                    #    - If Soft Restart was intended (L4), we MUST reset current_best to the new value.
                    #      Otherwise, the worker will be "forever stagnant" because it can't beat its old ghost.
                    #      BUT, we should NOT reset the *Strategy Level* if we want to count this as a failure attempt.
                    #    - Wait... if we reset current_best, the system thinks we found a "New Local Best" as soon as we climb 0.001.
                    #      This triggers "NEW LOCAL BEST" logic which resets stagnation_level = 0.
                    #      So, Soft Restart effectively resets the clock.
                    #
                    #    - How to allow L5 then?
                    #      L5 requires reaching L4, trying X times, and failing.
                    #      If Soft Restart (L4) resets current_best -> finds "improvement" -> resets stagnation -> Cycle L0...L4.
                    #
                    #    - SOLUTION: 
                    #      If we are "Attacking Global Best" (current_best >= global_best), we MUST NOT RESET current_best downward.
                    #      We must keep the high bar. If Soft Restart spawns us at 0.9*Best, and we climb to 0.95*Best,
                    #      that is NOT a success if our goal is > 1.0*Best.
                    #      So, for the Leader (Attacking Global Best), keep current_best high.
                    #      For a Follower (Local Optima), resetting is fine.
                    
                    if is_attacking_global_best:
                         self.logger("Leader Mode: Retaining high 'current_best' baseline to force meaningful improvement or hard restart trigger.")
                         # Do not reset current_best.
                         # Do not reset stagnation_level.
                         pass
                    else:
                         # Follower Mode: Reset and try somewhere else
                         self.logger("Follower Mode: Resetting 'current_best' to allow local hill climbing.")
                         current_best = env.key_value
                         # Resetting stagnation level here makes sense for followers - they just want to work.
                         # But if we want L4 to retry specifically... 
                         # Actually, if a follower restarts, they are effectively a new worker. Reset is fine.
                         self.stagnation_level = 0
                         self.phase_retries = 0

                
                # Reset counter to give the new candidate a chance
                no_improve_steps = 0
                
            # [NEW] Periodic Active Path Relinking to bridge peaks
            # Increase frequency from 300 to 100 to force more hybridization
            if self.current_run_steps % 100 == 0:
                 self.logger(f"Step:{self.current_run_steps} Cur:{env.key_value} Best:{current_best} (BK:{env.best_known}) Stagnation:{no_improve_steps}")
            
            # [NEW] Periodic Active Path Relinking to bridge peaks
            # Increase frequency from 300 to 100 to force more hybridization
            # [FIX 2026-02-19] IMMUNITY PERIOD: Do NOT relink if recently restarted (within 500 steps).
            # This allows new constructive solutions to mature without being pulled back to the black hole.
            if self.current_run_steps % 100 <= 1 and len(self.elite_pool) >= 2:
                 if (self.current_run_steps - self.last_restart_step) > 500:
                     self._apply_breakout(env, "active_pool_relinking")
                 else:
                     pass # Immunized
                 
                 # Code Check Fix: Do NOT reset no_improve_steps here. 
                 # We want the main stagnation logic (Heavy Ruin/Supernova) to still trigger if this fails.
                 # no_improve_steps = 0 
                 continue

            # Sync Distributed Elite Pool periodically
            if self.current_run_steps % 50 == 0:
                self._sync_shared_pool()
                
                # Check for Rebuild Signal from Follower Logic
                if self.pending_rebuild:
                    self.logger("Detected Rebuild Signal during Sync. Aborting current run... ")
                    break 
                
                # Check if we are currently in a "Recovery/Exploration" phase (high no_improve_steps)
                # If we just performed a massive ruin/injection, we need time to climb back up.
                # Don't kill promising young solutions too early.
                # Only apply catch-up if we have been stagnant for a while OR if the current solution is truly abysmal for too long.
                
                # [FIX 2026-02-20] Dynamic Immunity for "Rebel" Workers (Soft Restarted)
                # If a worker recently restarted, it enters "Exploration Mode".
                # We grant it a generous grace period (e.g. 2 * node_num steps or static 2000) to find a NEW peak.
                # During this time, it is immune to "Catch-up" (being pulled back to the old peak).
                
                node_num = env.instance_data["node_num"]
                immunity_period = max(2000, node_num) # Adaptive: At least 2000, or 1x node count
                is_immune = (self.current_run_steps - self.last_restart_step) < immunity_period

                if self.elite_pool and no_improve_steps > 300:
                    pool_best_wrapper = max(self.elite_pool, key=lambda s: s["solution"].cut_value)
                    pool_best = pool_best_wrapper["solution"]
                    
                    # Original: 0.998 allowed 5321 (0.9989) to survive indefinitely.
                    # Fix: 0.9992 was too strict and caused "Ruin -> Catch-up -> Reset" loop.
                    # Relaxed to 0.90 to allow deep exploration/ruin strategies to work.
                    catch_up_threshold = 0.90 
                    
                    # Logically: If NOT immune AND score is too low -> Catch up
                    if not is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        self.logger(f"AGGRESSIVE CATCH-UP: Abandoning {env.key_value} for {pool_best.cut_value} (Threshold: {catch_up_threshold})...")
                        
                        # Restore using standard wrapper interface
                        env.import_solution_wrapper(pool_best_wrapper)

                        
                        current_best = env.key_value
                        no_improve_steps = 0
                    elif is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        # Log sparsely
                        if self.current_run_steps % 500 == 0:
                             self.logger(f"Catch-up IMMUNITY: Worker exploring ({self.current_run_steps - self.last_restart_step}/{immunity_period} steps). Val={env.key_value}")



        return True
