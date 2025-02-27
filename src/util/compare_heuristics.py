import concurrent
import dill
import multiprocessing
import multiprocessing.managers
import azure.identity
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


def run_random_hh(
        env_serialized: bytes,
        running_heuristic_pool: list[str],
        max_steps: int,
        problem: str,
        best_result_proxy: multiprocessing.managers.ValueProxy=None,
        dump_best_result: bool=False
) -> float:
    
    random_hh = RandomHyperHeuristic(running_heuristic_pool, problem)
    env = dill.loads(env_serialized)
    complete_and_valid_solution = random_hh.run(env, max_steps=max_steps)

    if complete_and_valid_solution:
        # If found best, save it
        if dump_best_result and best_result_proxy:
            if best_result_proxy.value == float('-inf') or env.compare(env.key_value, best_result_proxy.value) >= 0:
                best_result_proxy.value = env.key_value
                env.dump_result(dump_trajectory=True, result_file=f"best_result_{env.key_value}.txt")
        return env.key_value
    else:
        return None

def evaluate_heuristic(
        env_serialized: bytes,
        heuristic_name: str,
        running_heuristic_pool: list[str],
        max_steps: int,
        search_interval: int,
        search_time: int,
        problem: str,
        best_result_proxy: multiprocessing.managers.ValueProxy=None,
        dump_best_result: bool=False,
) -> tuple[float, str, bytes]:
    env = dill.loads(env_serialized)
    heuristic = load_heuristic(heuristic_name, problem)
    operators = []
    for _ in range(search_interval):
        if env.continue_run:
            operators.append(env.run_heuristic(heuristic))
    after_step_env_serialized = dill.dumps(env)
    # MCTS to evaluate heuristic performance
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = [executor.submit(run_random_hh, after_step_env_serialized, running_heuristic_pool, max_steps, problem, best_result_proxy, dump_best_result) for _ in range(search_time)]
    for future in concurrent.futures.as_completed(future_results):
        result = future.result()
        if result:
            results.append(result)
    return heuristic_name, results, after_step_env_serialized, operators

def compare_heuristics(
        env: BaseEnv,
        candidate_heuristics: list[str],
        running_heuristic_pool: list[str],
        max_steps: int,
        search_interval: int,
        search_time: int,
        problem: str,
        best_result_proxy: multiprocessing.managers.ValueProxy=None,
        dump_best_result: bool=False
) -> tuple[str, bytes]:
    env_serialized = dill.dumps(env)
    futures = []
    for heuristic in candidate_heuristics:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures.append(executor.submit(
                evaluate_heuristic,
                env_serialized,
                heuristic,
                running_heuristic_pool,
                max_steps,
                search_interval,
                search_time,
                problem,
                best_result_proxy,
                dump_best_result
            ))

    total_results = []
    for future in concurrent.futures.as_completed(futures):
        heuristic_name, results, after_step_env_serialized, operators = future.result()
        total_results.append([heuristic_name, results, dill.loads(after_step_env_serialized), operators])
    return total_results