import os
import sys
import psutil
import multiprocessing
import random
import time
import platform
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.problems.max_cut.env import Env
from src.pipeline.hyper_heuristics.random_search_best import RandomSearchBestHyperHeuristic
from src.pipeline.hyper_heuristics.phased_search_best import PhasedSearchBestHyperHeuristic
from src.pipeline.hyper_heuristics.phased_search_best_ucb import PhasedSearchUCBBestHyperHeuristic
from src.pipeline.hyper_heuristics.phased_search_best_fast_stop import PhasedSearchFastStopBestHyperHeuristic
from src.pipeline.hyper_heuristics.phased_search_adaptive_polishing import PhasedSearchAdaptivePolishingHyperHeuristic
from src.pipeline.hyper_heuristics.phased_search_cooperative import PhasedSearchCooperativeHyperHeuristic
from src.util.filter_diverse_elites import get_diverse_elites



def log_system_status(context: str, logger=None):
    if logger is None:
        return
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        disk_info = f"Disk R/W: {disk.read_bytes>>20}MB/{disk.write_bytes>>20}MB" if disk else "Disk: N/A"
        load_avg = "N/A"
        if hasattr(os, 'getloadavg'):
            load_avg = f"{os.getloadavg()}"
            
        logger(f"[System Status - {context}] Host: {platform.node()} | CPU: {cpu_percent}% | Load: {load_avg} | "
              f"Mem: {mem.percent}% (Used: {mem.used>>20}MB, Avail: {mem.available>>20}MB) | {disk_info}")
    except Exception as e:
        logger(f"Failed to log system status: {e}")


def _probe_env_mem(data_name: str, heuristic_dir: str) -> int:
    import os, time, random
    import psutil
    try:
        import numpy as np
    except Exception:
        np = None

    seed = time.time_ns() ^ os.getpid() ^ int.from_bytes(os.urandom(8), 'little')
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xFFFFFFFF)

    env = Env(data_name=data_name)
    construction_steps = env.construction_steps
    env.reset()
    heuristic_pool = [os.path.join(heuristic_dir, f) for f in os.listdir(heuristic_dir) if f.endswith(".py")]
    algorithm = RandomSearchBestHyperHeuristic(heuristic_pool, "max_cut", 2)

    rss = psutil.Process(os.getpid()).memory_info().rss

    try:
        del env, algorithm
        import gc; gc.collect()
    except Exception:
        pass
    return rss, construction_steps

def pick_safe_workers(data_name: str, heuristic_dir: str,
                      safety_factor: float = 1.5,
                      reserve_fraction: float = 0.2) -> int:
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(1) as pool:
        mem_per_task, construction_steps = pool.apply(_probe_env_mem, (data_name, heuristic_dir))

    avail = psutil.virtual_memory().available
    budget = int(avail * (1.0 - reserve_fraction))
    max_by_mem = max(1, budget // int(mem_per_task * safety_factor))

    max_by_cpu = os.cpu_count() or 1
    # Remove artificial cap of 24 workers. Let hardware decide.
    workers = max(1, min(max_by_cpu, max_by_mem))

    # print(f"Estimated per-task RSS ~ {mem_per_task/1024/1024:.1f} MiB, avail ~ {avail/1024/1024:.1f} MiB, choose workers={workers}", flush=True)

    return workers, mem_per_task, avail

def run_once(
        data_name: str,
        heuristic_dir: str,
        experiment_dir: str,
        run_id: int,
        method: str = "phased",
        shared_pool_dir: str = None,
        log_file_path: str = None
) -> float:
    try:
        seed = time.time_ns() ^ os.getpid() ^ int.from_bytes(os.urandom(8), 'little')
    except Exception:
        seed = time.time_ns() ^ os.getpid()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xFFFFFFFF)

    env = Env(data_name=data_name)

    env.reset(output_dir=os.path.join(experiment_dir, "result"))
    
    # Use absolute paths for heuristics to avoid ambiguity
    heuristic_pool = [os.path.join(heuristic_dir, f) for f in os.listdir(heuristic_dir) if f.endswith(".py")]

    # Local logger that writes ONLY to file
    def local_log(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{timestamp}, Worker:{run_id}] {message}"
        if log_file_path:
            try:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass
    
    log_system_status(f"Worker:{run_id} Start", logger=local_log)
    
    if method == "phased":
        algorithm = PhasedSearchBestHyperHeuristic(heuristic_pool, "max_cut")
    elif method == "ucb":
        algorithm = PhasedSearchUCBBestHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            shared_pool_dir=shared_pool_dir,
            top_k=10,
            load_ratio=1.0
        )
    elif method == "fast_stop":
        algorithm = PhasedSearchFastStopBestHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            shared_pool_dir=shared_pool_dir,
            top_k=10,
            load_ratio=1.0,
            fail_fast_threshold=0.02
        )
    elif method == "adaptive_polishing":
        algorithm = PhasedSearchAdaptivePolishingHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            shared_pool_dir=shared_pool_dir,
            top_k=10,
            load_ratio=1.0,
            fail_fast_threshold=0.02
        )
    elif method == "cooperative":
        algorithm = PhasedSearchCooperativeHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            shared_pool_dir=shared_pool_dir,
            worker_id=run_id,
            logger=local_log
        )
    elif method == "random":
        algorithm = RandomSearchBestHyperHeuristic(heuristic_pool, "max_cut", iterations_scale_factor=50)
        
    algorithm.run(env)

def main(
        data_name: str,
        heuristic_dir: str,
        num_runs: int,
        method: str = "phased",
        experiment_name: str = None
    ):
    workers, mem_per_task, avail = pick_safe_workers(data_name, heuristic_dir)
        
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")

    if num_runs is None:
        num_runs = workers

    remaining = list(range(num_runs))
    finished_ids = []
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"

    if experiment_name is None: 
        experiment_dir = os.path.join(base_output_dir, "max_cut", data_name)
    else:
        experiment_dir = os.path.join(base_output_dir, "max_cut", experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    log_file_path = os.path.join(experiment_dir, "run.log")

    # Main logger that writes ONLY to file
    def main_logger(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{timestamp}, Main] {message}"
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
                f.flush()
                # Force OS to write to disk
                os.fsync(f.fileno())
        except Exception:
            pass
    
    logger = main_logger
    logger(f"Starting {method} Search for {data_name} with {workers} workers. Output: {experiment_dir}")
    logger(f"Log file: {log_file_path}")
    if num_runs == workers: # Originally "is None" but now we check if it was defaulted
        logger(f"Num runs not specified. Defaulting to max capacity: {workers}")
    logger(f"Estimated per-task RSS ~ {mem_per_task/1024/1024:.1f} MiB, avail ~ {avail/1024/1024:.1f} MiB")

    # [INFO] Print Problem Statistics ONCE at Startup
    temp_env = Env(data_name=data_name)
    node_num = temp_env.instance_data.get("node_num", "Unknown")
    bk = temp_env.best_known
    logger(f"============================================================")
    logger(f"  Target Data: {data_name}")
    logger(f"  Nodes: {node_num}")
    logger(f"  Best Known (BK): {bk}")
    logger(f"============================================================")

    if method == "cooperative":
        # Auto-configure shared pool directory for cooperative methods (communication channel)
        # Shared pool is still tied to the experiment directory to keep runs isolated if needed
        shared_pool_dir = os.path.join(experiment_dir, "elite_pool")
        os.makedirs(shared_pool_dir, exist_ok=True)
        logger(f"Shared Elite Pool: {shared_pool_dir}")

    log_system_status("Main Start", logger=logger)

    while remaining:
        logger(f"Start batch with workers={workers}, remaining tasks={len(remaining)}")
        log_system_status(f"Batch Start (Remaining: {len(remaining)})", logger=logger)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            fut_map = {executor.submit(
                run_once, 
                data_name, 
                heuristic_dir, 
                experiment_dir, 
                run_id, 
                method=method, 
                shared_pool_dir=shared_pool_dir,
                log_file_path=log_file_path
            ): run_id for run_id in remaining}

            for fut in as_completed(fut_map):
                run_id = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger(f"Run {run_id} failed: {e}")
                finished_ids.append(run_id)

        done_ids = {run_id for run_id in finished_ids}
        remaining = [rid for rid in remaining if rid not in done_ids]

        if remaining:
            workers = max(1, workers // 2)
            time.sleep(1.0)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyper-heuristic search for MaxCut")
    parser.add_argument("data_name", type=str, help="Name of the dataset (e.g., g1)")
    parser.add_argument("-n", "--num_runs", type=int, default=None, help="Number of parallel runs (default: max capable)")
    parser.add_argument("-d", "--heuristic_dir", type=str, 
                        default="evolved_heuristics.part3", help="Directory containing heuristics")
    parser.add_argument("-m", "--method", type=str, default="cooperative", choices=["phased", "random", "ucb", "fast_stop", "adaptive_polishing", "cooperative"], 
                        help="Search method: 'phased', 'random', 'ucb', 'fast_stop', 'adaptive_polishing', or 'cooperative' (default: fast_stop)")
    parser.add_argument("-exp", "--experiment_name", type=str, default=None, help="Experiment name (default: None, uses data_name)")

    args = parser.parse_args()
    main(args.data_name, os.path.join("src", "problems", "max_cut", "heuristics", args.heuristic_dir), args.num_runs, args.method, args.experiment_name)