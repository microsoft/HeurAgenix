import os
import psutil
import multiprocessing
import random
import time
import numpy as np
import importlib
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.util.logger import build_logger, log_system_status

def _probe_env_mem(problem: str, data_name: str, heuristic_dir: str) -> int:
    from src.common.hyper_heuristics.random_search_best import RandomSearchBestHyperHeuristic
    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")

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
    algorithm = RandomSearchBestHyperHeuristic(heuristic_pool, problem, 2)

    rss = psutil.Process(os.getpid()).memory_info().rss

    try:
        del env, algorithm
        import gc; gc.collect()
    except Exception:
        pass
    return rss, construction_steps

def pick_safe_workers(problem: str, data_name: str, heuristic_dir: str,
                      safety_factor: float = 1.5,
                      reserve_fraction: float = 0.2) -> int:
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(1) as pool:
        mem_per_task, construction_steps = pool.apply(_probe_env_mem, (problem, data_name, heuristic_dir))

    avail = psutil.virtual_memory().available
    budget = int(avail * (1.0 - reserve_fraction))
    max_by_mem = max(1, budget // int(mem_per_task * safety_factor))

    max_by_cpu = os.cpu_count() or 1
    workers = max(1, min(max_by_cpu, max_by_mem))


    return workers, mem_per_task, avail

def run_once(
        problem: str,
        data_name: str,
        heuristic_dir: str,
        experiment_dir: str,
        run_id: int,
        method: str = "phased",
        shared_pool_dir: str = None,
        log_file_path: str = None,
        max_restarts: int = None
) -> float:
    try:
        seed = time.time_ns() ^ os.getpid() ^ int.from_bytes(os.urandom(8), 'little')
    except Exception:
        seed = time.time_ns() ^ os.getpid()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xFFFFFFFF)

    # Dynamic environment loading
    env_module = importlib.import_module(f"src.problems.{problem}.env")
    EnvClass = getattr(env_module, "Env")
    env = EnvClass(data_name=data_name)

    env.reset(output_dir=os.path.join(experiment_dir, "result"))
    
    # Use absolute paths for heuristics to avoid ambiguity
    heuristic_pool = [os.path.join(heuristic_dir, f) for f in os.listdir(heuristic_dir) if f.endswith(".py")]

    # Local logger using the shared build_logger
    local_log = build_logger(log_file_path, f"Worker:{run_id}")
    
    log_system_status(f"Worker:{run_id} Start", logger=local_log)
    
    hh_name = method
    try:
        module = importlib.import_module(f"src.problems.{problem}.hyper_heuristics.{hh_name}")
    except ImportError:
        try:
            module = importlib.import_module(f"src.common.hyper_heuristics.{hh_name}")
        except ImportError:
            module = None
    if module is None:
        local_log(f"Error: Could not load hyper-heuristic module for '{method}' (tried problem-specific and common paths)")
        return 0

    # Resolve Class Name
    # Default Rule: snake_case -> CamelCase + "HyperHeuristic"
    class_name = "".join(x.title() for x in hh_name.split("_")) + "HyperHeuristic"
    
    hh_class = getattr(module, class_name, None)
    
    # Fallback Rule: Search for any class ending in "HyperHeuristic"
    if hh_class is None:
        classes = [obj for name, obj in inspect.getmembers(module) 
                   if inspect.isclass(obj) and name.endswith("HyperHeuristic")
                   and obj.__module__ == module.__name__] # Ensure defined in module, not imported
        if len(classes) == 1:
            hh_class = classes[0]
        elif len(classes) > 1:
             # Try loose match
             norm_name = hh_name.replace("_", "").lower()
             for cls in classes:
                 if norm_name in cls.__name__.lower():
                     hh_class = cls
                     break
                     
    if hh_class is None:
        local_log(f"Error: Could not find HyperHeuristic class in {module.__name__}")
        return 0.0

    # Construct Arguments
    kwargs = {
        "heuristic_pool": heuristic_pool,
        "problem": problem,
        "shared_pool_dir": shared_pool_dir,
        "worker_id": run_id,
        "max_restarts": max_restarts,
        "logger": local_log,
        "top_k": 10,
        "load_ratio": 1.0,
        "fail_fast_threshold": 0.02
    }

    try:
        algorithm = hh_class(**kwargs)
    except Exception as e:
        local_log(f"Error instantiating {hh_class.__name__}: {e}")
        return 0.0
    algorithm.run(env)

def main(
        problem: str,
        data_name: str,
        heuristic_dir: str,
        num_runs: int,
        method: str = "phased",
        experiment_name: str = None,
        max_restarts: int = None
    ):
    workers, mem_per_task, avail = pick_safe_workers(problem, data_name, heuristic_dir)
        
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")

    if num_runs is None:
        num_runs = workers
    else:
        # Respect user-specified run count as a hard cap on concurrent workers.
        workers = max(1, min(workers, int(num_runs)))

    remaining = list(range(num_runs))
    finished_ids = []
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"

    if experiment_name is None: 
        experiment_dir = os.path.join(base_output_dir, problem, data_name)
    else:
        experiment_dir = os.path.join(base_output_dir, problem, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    log_file_path = os.path.join(experiment_dir, "run.log")

    # Main logger using the shared build_logger
    logger = build_logger(log_file_path, "Main")
    
    logger(f"Starting {method} Search for {data_name} with {workers} workers. Output: {experiment_dir}")
    logger(f"Log file: {log_file_path}")
    if num_runs == workers: # Originally "is None" but now we check if it was defaulted
        logger(f"Num runs not specified. Defaulting to max capacity: {workers}")
    logger(f"Estimated per-task RSS ~ {mem_per_task/1024/1024:.1f} MiB, avail ~ {avail/1024/1024:.1f} MiB")

    # [INFO] Print Problem Statistics ONCE at Startup
    env_module = importlib.import_module(f"src.problems.{problem}.env")
    EnvClass = getattr(env_module, "Env")
    env = EnvClass(data_name=data_name)
    bk = env.best_known
    logger(f"=" * 50)
    logger(f"  Problem Name: {problem}")
    logger(f"  Heuristic Directory: {heuristic_dir}")
    logger(f"  Method: {method}")
    logger(f"  Experiment Name: {experiment_name}")
    logger(f"  Target Data: {data_name}")
    for key, value in env.instance_data.items():
        try:
            val_str = str(value)
            if len(val_str) < 100:
                logger(f"  {key}: {val_str}")
        except:
            pass
    logger(f"  Best Known (BK): {bk}")
    logger(f"=" * 50)

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
                problem,
                data_name, 
                heuristic_dir, 
                experiment_dir, 
                run_id, 
                method=method, 
                shared_pool_dir=shared_pool_dir,
                log_file_path=log_file_path,
                max_restarts=max_restarts
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
    parser.add_argument("-p", "--problem", choices=["max_cut", "cvrp"], default="max_cut", help="Specifies the type of combinatorial optimization problem.")
    parser.add_argument("-n", "--num_runs", type=int, default=None, help="Number of parallel runs (default: max capable)")
    parser.add_argument("-d", "--heuristic_dir", type=str, 
                        default="evolved_heuristics.part3", help="Directory containing heuristics")
    parser.add_argument("-m", "--method", type=str, default="island_vnd_discrete", help="Hyper heuristics method")
    parser.add_argument("-exp", "--experiment_name", type=str, default=None, help="Experiment name (default: None, uses data_name)")
    parser.add_argument("-r", "--max_restarts", type=int, default=0, help="Maximum number of global restarts before exiting worker")

    args = parser.parse_args()
    main(
        args.problem,
        args.data_name,
        os.path.join("src", "problems", args.problem, "heuristics", args.heuristic_dir),
        args.num_runs,
        args.method,
        args.experiment_name,
        args.max_restarts
    )