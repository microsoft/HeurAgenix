import os
import importlib
import copy
import numpy as np
import inspect
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.util.util import load_heuristic


def data_collection(
        problem: str,
        data_path: str,
        score_calculation: callable,
        prune_frequency: int=200,
        prune_ratio: float=1.02,
        heuristic_dir: str=None,
        output_dir: str=None,
        search_time: int=1000
) -> None:
    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")
    env = Env(data_path, mode="test")

    data_name = data_path.split(os.sep)[-1]
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics") if heuristic_dir is None else heuristic_dir
    heuristic_name_str = ", ".join([heuristic_file.split(".")[0] for heuristic_file in os.listdir(heuristic_dir)])
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
    information_file.write(f"search_time: {search_time}\n")
    information_file.close()
    
    # TODO:

    return

    random_hh = RandomHyperHeuristic(heuristic_dir, problem=problem)

    selected_previous_heuristics = []
    for round in range(int(env.construction_steps * 2)):
        best_score = np.inf
        for heuristic in os.listdir(heuristic_dir):
            env.reset(f"round_{round}_{heuristic}")
            for h in selected_previous_heuristics:
                env.run_heuristic(load_heuristic(h, heuristic_dir))
            env.run_heuristic(load_heuristic(heuristic, heuristic_dir))
            saved_solution = copy.deepcopy(env.current_solution)
            saved_state = copy.deepcopy(env.state_data)
            saved_algorithm_data = copy.deepcopy(env.algorithm_data)

            results = []
            for _ in range(search_time):
                env.current_solution = saved_solution
                env.state_data = saved_state
                env.algorithm_data = saved_algorithm_data
                random_hh.run(env)
                results.append(env.key_value)
                if (_ + 1) % 200 == 0:

                    sorted(results)
                    results = sorted(results)
                    average = sum(results) / len(results)
                    top_k_average = sum(results[:top_k]) / top_k
                    score = r * average + (1 - r) * top_k_average
                    if score > best_score * 1.01:
                        break
            results = sorted(results)
            average = sum(results) / len(results)
            top_k_average = sum(results[:top_k]) / top_k
            score = r * average + (1 - r) * top_k_average
            # r *= 1.01
            output_file.write(heuristic + ", " + str(_) + ", " + str(average) + ", " + str(top_k_average) + ", " + str(score) + ", " + str(results) + "\n")
            if score < best_score:
                best_score = score
                best_heuristics = heuristic
        selected_previous_heuristics.append(best_heuristics)
        output_file.write(best_heuristics + "\n---------------\n")
    output_file.close()

if __name__ == "__main__":
    score_calculation = lambda results: 0.8 * sum(results) / len(results) + 0.2 * sum(sorted(results[: 20])) / 20
    data_collection(
        problem="tsp",
        data_path="bayg29.tsp",
        score_calculation=score_calculation,
        prune_frequency=200,
        prune_ratio=1.02,
        search_time=1000
    )