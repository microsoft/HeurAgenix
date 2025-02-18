import concurrent
import dill
import multiprocessing
import multiprocessing.managers
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic


def run_random_hh(
        env_serialized: bytes,
        running_heuristic_pool: list[str],
        max_steps: int,
        best_result_proxy: multiprocessing.managers.ValueProxy,
        problem: str,
        dump_best_result: bool=False
) -> float:
    
    random_hh = RandomHyperHeuristic(running_heuristic_pool, problem)
    env = dill.loads(env_serialized)
    complete_and_valid_solution = random_hh.run(env, max_steps=max_steps)

    if complete_and_valid_solution:
        # If found best, save it
        if dump_best_result and best_result_proxy.value == float('-inf') or env.compare(env.key_value, best_result_proxy.value) >= 0:
            best_result_proxy.value = env.key_value
            env.dump_result(True, "best_result.txt")
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
        best_result_proxy: multiprocessing.managers.ValueProxy,
        problem: str,
        dump_best_result: bool=False,
) -> tuple[float, str, bytes]:
    env = dill.loads(env_serialized)
    heuristic = load_heuristic(heuristic_name, problem)
    for _ in range(search_interval):
        env.run_heuristic(heuristic)
    after_step_env_serialized = dill.dumps(env)
    # MCTS to evaluate heuristic performance
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = [executor.submit(run_random_hh, after_step_env_serialized, running_heuristic_pool, max_steps, best_result_proxy, problem, dump_best_result) for _ in range(search_time)]
    for future in concurrent.futures.as_completed(future_results):
        result = future.result()
        if result:
            results.append(result)
    return results, after_step_env_serialized

def compare_heuristics(
        env: BaseEnv,
        candidate_heuristics: list[str],
        running_heuristic_pool: list[str],
        max_steps: int,
        search_interval: int,
        search_time: int,
        best_result_proxy: multiprocessing.managers.ValueProxy,
        problem: str,
        dump_best_result: bool=False
) -> tuple[str, bytes]:
    best_average_score = None
    best_after_heuristic_env = None
    best_heuristic = None
    env_serialized = dill.dumps(env)
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_heuristic, env_serialized, heuristic, running_heuristic_pool, max_steps, search_interval, search_time, best_result_proxy, problem, dump_best_result) for heuristic in candidate_heuristics]

    for heuristic_index, future in enumerate(concurrent.futures.as_completed(futures)):
        results, after_step_env_serialized = future.result()
        average_score = None if len(results) <= 0 else sum(results) / len(results)
        if average_score is not None and best_average_score is None or env.compare(average_score, best_average_score) > 0:
            best_average_score = average_score
            best_after_heuristic_env = after_step_env_serialized
            best_heuristic = candidate_heuristics[heuristic_index]
    return best_heuristic, dill.loads(best_after_heuristic_env)