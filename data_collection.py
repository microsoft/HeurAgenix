import os
import importlib
import copy
import numpy as np
import inspect
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.util.util import load_heuristic


def filter_deterministic_heuristics(heuristic_dir: str, env: object) -> list[str]:
    previous_steps = int(env.construction_steps / 5)
    try_steps = int(env.construction_steps / 5)
    try_times = 10

    deterministic_heuristics_names = []
    random_heuristics_names = []
    env.reset()
    RandomHyperHeuristic(problem=problem, heuristic_dir=heuristic_dir).run(env, previous_steps)
    saved_solution = copy.deepcopy(env.current_solution)
    saved_state = copy.deepcopy(env.state_data)
    saved_algorithm_data = copy.deepcopy(env.algorithm_data)

    for heuristic_file in os.listdir(heuristic_dir):
        heuristic = load_heuristic(heuristic_file, heuristic_dir)
        key_value = None
        deterministic_flag = True
        for _ in range(try_times):
            env.current_solution = saved_solution
            env.state_data = saved_state
            env.algorithm_data = saved_algorithm_data
            for _ in range(try_steps):
                env.run_heuristic(heuristic)
            if key_value == None:
                key_value = env.key_value
            if key_value != env.key_value:
                deterministic_flag = False
                break

        if deterministic_flag:
            deterministic_heuristics_names.append(heuristic_file.split(".")[0])
        else:
            random_heuristics_names.append(heuristic_file.split(".")[0])
    return deterministic_heuristics_names, random_heuristics_names

def data_collection(
        problem: str,
        data_path: str,
        score_calculation: callable,
        prune_frequency: int=200,
        prune_ratio: float=1.02,
        heuristic_dir: str=None,
        search_time: int=1000,
        output_dir: str=None
) -> None:
    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")
    env = Env(data_path, mode="test")

    data_name = data_path.split(os.sep)[-1]
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics") if heuristic_dir is None else heuristic_dir
    heuristic_name_str = ",".join([heuristic_file.split(".")[0] for heuristic_file in os.listdir(heuristic_dir)])
    deterministic_heuristics_names, random_heuristics_names = filter_deterministic_heuristics(heuristic_dir, env)
    deterministic_heuristic_name_str = ",".join(deterministic_heuristics_names)
    random_heuristic_name_str = ",".join(random_heuristics_names)
    output_dir = os.path.join("output", problem, "data_collection", f"{data_name}.{datetime_str}.result") if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir is {output_dir}")

    information_file = open(os.path.join(output_dir, "information.txt"), "w")
    information_file.write(f"problem: {problem}\n")
    information_file.write(f"data_path: {data_path}\n")
    information_file.write(f"score_calculation: \n{inspect.getsource(score_calculation)}\n")
    information_file.write(f"prune_frequency: {prune_frequency}\n")
    information_file.write(f"prune_ratio: {prune_ratio}\n")
    information_file.write(f"heuristic_dir: {heuristic_dir}\n")
    information_file.write(f"heuristics: {heuristic_name_str}\n")
    information_file.write(f"deterministic_heuristics: {deterministic_heuristic_name_str}\n")
    information_file.write(f"random_heuristics: {random_heuristic_name_str}\n")
    information_file.write(f"search_time: {search_time}\n")
    information_file.close()
    
    random_hh = RandomHyperHeuristic(problem=problem, heuristic_names=deterministic_heuristics_names, heuristic_dir=heuristic_dir)

    selected_previous_heuristics = []
    for round_index in range(int(env.construction_steps * 2)):
        env.reset()
        selected_previous_heuristics_str = ""
        for heuristic_name in selected_previous_heuristics:
            operator = env.run_heuristic(load_heuristic(heuristic_name, heuristic_dir))
            selected_previous_heuristics_str += f"{heuristic_name}, {operator}\n"
        saved_solution = copy.deepcopy(env.current_solution)
        saved_state = copy.deepcopy(env.state_data)
        saved_algorithm_data = copy.deepcopy(env.algorithm_data)

        output_file = open(os.path.join(output_dir, f"round_{round_index}.txt"), "w")
        output_file.write(f"selected_previous_heuristics, operators: \n{selected_previous_heuristics_str}\n")
        output_file.write(f"current_solution: \n{saved_solution}\n")

        best_score = np.inf
        output_file.write("---------------\n")
        output_file.write(f"heuristic\tscore\tresults\n")
        for heuristic_file in os.listdir(heuristic_dir):
            env.reset()
            heuristic_name = heuristic_file.split(".")[0]

            results = []
            for search_index in range(search_time):
                env.current_solution = saved_solution
                env.state_data = saved_state
                env.algorithm_data = saved_algorithm_data
                env.run_heuristic(load_heuristic(heuristic_file, heuristic_dir))
                random_hh.run(env)
                results.append(env.key_value)
                if (search_index + 1) % prune_ratio == 0:
                    env.compare(best_score * prune_ratio, score_calculation(results)) > 0
                    break
            score = score_calculation(results)
            results_str = ",".join([str(result) for result in sorted(results)])
            output_file.write(f"{heuristic_name}\t{score}\t{results_str}\n")

            if env.compare(score, best_score) > 0:
                best_score = score
                best_heuristics = heuristic_name
        selected_previous_heuristics.append(best_heuristics)
        output_file.write("---------------\n")
        output_file.write(f"best_heuristics:\t{best_heuristics}")
        output_file.close()

if __name__ == "__main__":
    problem = "tsp"
    data_name = "bayg29.tsp"
    score_calculation = lambda results: 0.8 * sum(results) / len(results) + 0.2 * sum(sorted(results[: 20])) / 20
    base_output_dir = os.getenv("AMLT_OUTPUT_DIR") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    prune_frequency = 200
    prune_ratio = 1.02
    search_time = 1000
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, problem, "data_collection", f"{data_name}.{datetime_str}.result")
    data_collection(
        problem=problem,
        data_path=data_name,
        score_calculation=score_calculation,
        prune_frequency=prune_frequency,
        prune_ratio=prune_ratio,
        search_time=search_time,
        output_dir=output_dir
    )