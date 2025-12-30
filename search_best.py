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


def log_system_status(context: str):
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        disk_info = f"Disk R/W: {disk.read_bytes>>20}MB/{disk.write_bytes>>20}MB" if disk else "Disk: N/A"
        load_avg = "N/A"
        if hasattr(os, 'getloadavg'):
            load_avg = f"{os.getloadavg()}"
            
        print(f"[System Status - {context}] Host: {platform.node()} | CPU: {cpu_percent}% | Load: {load_avg} | "
              f"Mem: {mem.percent}% (Used: {mem.used>>20}MB, Avail: {mem.available>>20}MB) | {disk_info}", flush=True)
    except Exception as e:
        print(f"Failed to log system status: {e}", flush=True)


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
    workers = max(1, min(max_by_cpu, max_by_mem))

    if construction_steps > 5000:
        workers = min(workers, 24)
    print(f"Estimated per-task RSS ~ {mem_per_task/1024/1024:.1f} MiB, avail ~ {avail/1024/1024:.1f} MiB, choose workers={workers}", flush=True)

    return workers

def run_once(data_name: str, heuristic_dir: str, experiment_dir: str, run_id: int, method: str = "phased", high_quality_solution_dir: str = None, top_k: int = 5, load_ratio: float = 0.8, fail_fast_threshold: float = 0.02) -> float:
    try:
        seed = time.time_ns() ^ os.getpid() ^ int.from_bytes(os.urandom(8), 'little')
    except Exception:
        seed = time.time_ns() ^ os.getpid()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xFFFFFFFF)

    env = Env(data_name=data_name)

    output_dir = os.path.join(experiment_dir, str(run_id))

    env.reset(output_dir=output_dir)
    
    log_system_status(f"Worker-{run_id} Start")
    
    # Use absolute paths for heuristics to avoid ambiguity
    heuristic_pool = [os.path.join(heuristic_dir, f) for f in os.listdir(heuristic_dir) if f.endswith(".py")]
    
    if method == "phased":
        algorithm = PhasedSearchBestHyperHeuristic(heuristic_pool, "max_cut")
    elif method == "ucb":
        algorithm = PhasedSearchUCBBestHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            high_quality_solution_dir=high_quality_solution_dir,
            top_k=top_k,
            load_ratio=load_ratio
        )
    elif method == "fast_stop":
        algorithm = PhasedSearchFastStopBestHyperHeuristic(
            heuristic_pool, 
            "max_cut", 
            high_quality_solution_dir=high_quality_solution_dir,
            top_k=top_k,
            load_ratio=load_ratio,
            fail_fast_threshold=fail_fast_threshold
        )
    elif method == "random":
        algorithm = RandomSearchBestHyperHeuristic(heuristic_pool, "max_cut", iterations_scale_factor=50)
        
    algorithm.run(env)
    return 

def main(
        data_name: str,
        heuristic_dir: str,
        num_runs: int,
        method: str = "phased",
        top_k: int = 5,
        load_ratio: float = 0.8,
        fail_fast_threshold: float = 0.02
    ):
    workers = pick_safe_workers(data_name, heuristic_dir)
        
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")

    remaining = list(range(num_runs))
    finished_ids = []
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_output_dir, "max_cut", f"search_best_result.{method}", data_name, experiment_name)
    
    high_quality_solution_dir = os.path.join(base_output_dir, "max_cut", f"search_best_result.{method}", data_name, "high_quality_solution")
    os.makedirs(high_quality_solution_dir, exist_ok=True)
    
    print(f"Starting {method} Search for {data_name} with {workers} workers. Output: {experiment_dir}")
    print(f"High Quality Solution Pool: {high_quality_solution_dir}")
    print(f"Cooperative Search: Top-K={top_k}, Load Ratio={load_ratio}")

    log_system_status("Main Start")

    while remaining:
        print(f"Start batch with workers={workers}, remaining tasks={len(remaining)}", flush=True)
        log_system_status(f"Batch Start (Remaining: {len(remaining)})")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            fut_map = {executor.submit(
                run_once, 
                data_name, 
                heuristic_dir, 
                experiment_dir, 
                run_id, 
                method=method, 
                high_quality_solution_dir=high_quality_solution_dir,
                top_k=top_k,
                load_ratio=load_ratio,
                fail_fast_threshold=fail_fast_threshold
            ): run_id for run_id in remaining}

            for fut in as_completed(fut_map):
                run_id = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
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
    parser.add_argument("-n", "--num_runs", type=int, default=100, help="Number of parallel runs (default: 100)")
    parser.add_argument("-d", "--heuristic_dir", type=str, 
                        default="evolved_heuristics.part3", help="Directory containing heuristics")
    parser.add_argument("-m", "--method", type=str, default="ucb", choices=["phased", "random", "ucb"], 
                        help="Search method: 'phased', 'random', or 'ucb' (default: phased)")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Number of top solutions to consider for loading (default: 10)")
    parser.add_argument("-r", "--load_ratio", type=float, default=0.8, help="Probability of loading an initial solution (default: 0.8)")
    parser.add_argument("-f", "--fail_fast_threshold", type=float, default=0.02, help="Fail fast threshold (default: 0.02)")


    args = parser.parse_args()
    main(args.data_name, os.path.join("src", "problems", "max_cut", "heuristics", args.heuristic_dir), args.num_runs, args.method, args.top_k, args.load_ratio, fail_fast_threshold=args.fail_fast_threshold)