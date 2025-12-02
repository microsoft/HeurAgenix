import os
import sys
import psutil
import multiprocessing
import random
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.problems.max_cut.env import Env
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime



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
    algorithm = RandomHyperHeuristic(os.listdir(heuristic_dir), "max_cut", 2)

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
    print(f"Estimated per-task RSS ~ {mem_per_task/1024/1024:.1f} MiB, "
          f"avail ~ {avail/1024/1024:.1f} MiB, choose workers={workers}")

    return workers

def run_once(data_name: str, heuristic_dir: str, run_id: int) -> float:
    try:
        seed = time.time_ns() ^ os.getpid() ^ int.from_bytes(os.urandom(8), 'little')
    except Exception:
        seed = time.time_ns() ^ os.getpid()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xFFFFFFFF)

    env = Env(data_name=data_name)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{datetime_str}_id_{run_id}"

    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, "max_cut", "search_best_result.update", env.data_ref_name, experiment_name)
    os.makedirs(os.path.join(base_output_dir, "max_cut", "search_best_result.update", env.data_ref_name), exist_ok=True)

    env.reset(output_dir=output_dir)
    print(f"Run id: {run_id}, seed: {seed}, output_dir: {output_dir}")
    algorithm = RandomHyperHeuristic(os.listdir(heuristic_dir), "max_cut", run_id, 5)
    found_best = algorithm.run(env)
    # env.dump_result(result_file="result.txt")
    print(f"Finish run id: {run_id}, env.key_value: {env.key_value}")
    return found_best, experiment_name

def main(data_name: str, heuristic_dir: str, num_runs: int):
    workers = pick_safe_workers(data_name, heuristic_dir)
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")

    remaining = list(range(num_runs))
    finished_ids = []

    while remaining:
        print(f"Start batch with workers={workers}, remaining tasks={len(remaining)}")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            fut_map = {executor.submit(run_once, data_name, heuristic_dir, run_id): run_id
                       for run_id in remaining}

            for fut in as_completed(fut_map):
                run_id = fut_map[fut]
                try:
                    found_best, experiment_name = fut.result()
                    finished_ids.append(run_id)
                    if found_best:
                        print(f"Run {run_id} found best solution in experiment {experiment_name}.")
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")

        done_ids = {run_id for run_id in finished_ids}
        remaining = [rid for rid in remaining if rid not in done_ids]

        if remaining:
            workers = max(1, workers // 2)
            time.sleep(1.0)
if __name__ == '__main__':
    data_name = sys.argv[1]
    num_runs = 100
    heuristic_dir = os.path.join("src", "problems", "max_cut", "heuristics", "evolved_heuristics.part2")
    if len(sys.argv) > 2:
        num_runs = int(sys.argv[2])
    if len(sys.argv) > 3:
        heuristic_dir = sys.argv[3]
    main(data_name, heuristic_dir, num_runs)
