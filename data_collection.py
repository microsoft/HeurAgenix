import argparse
import os
import copy
import importlib
import inspect
import numpy as np
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.util.util import load_heuristic

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path for source data.")
    parser.add_argument("-e", "--heuristic_dir", default=None, help="Path of heuristic dir.")
    parser.add_argument("-s", "--search_time", default=1000, type=int, help="MCTS times for each heuristic.")    
    parser.add_argument("-sc", "--score_calculation", choices=["average_score", "a8t2"], default="average_score", help="Function to calculate score.")
    parser.add_argument("-pf", "--prune_frequency", default=200, help="Prune and early stop frequency.")
    parser.add_argument("-pr", "--prune_ratio", default=1.02, help="Prune and early stop threshold.")
    parser.add_argument("-o", "--output_dir", default="output", help="Path of output dir")

    return parser.parse_args()

def a8t2(results: list[float]) -> float:
    top_k = 20
    average_score_ratio = 0.8
    average_score = sum(results) / len(results)
    top_k_score = sum(sorted(results[: top_k])) / top_k
    score = average_score_ratio * average_score + (1 - average_score_ratio) * top_k_score
    return score

def average_score(results: list[float]) -> float:
    average_score = sum(results) / len(results)
    return average_score


def filter_deterministic_heuristics(problem: str, heuristic_dir: str, env: object) -> list[str]:
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
        for try_index in range(try_times):
            env.current_solution = saved_solution
            env.state_data = saved_state
            env.algorithm_data = saved_algorithm_data
            for step_index in range(try_steps):
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
        heuristic_dir: str=None,
        score_calculation: callable=average_score,
        prune_frequency: int=200,
        prune_ratio: float=1.02,
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
    deterministic_heuristics_names, random_heuristics_names = filter_deterministic_heuristics(problem, heuristic_dir, env)
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
    previous_operator = []
    for round_index in range(int(env.construction_steps * 2)):
        env.reset()
        selected_previous_heuristics_str = ""
        for previous_step in range(len(selected_previous_heuristics)):
            heuristic_name = selected_previous_heuristics[previous_step]
            operator = previous_operator[previous_step]
            env.run_operator(operator)
            selected_previous_heuristics_str += f"{heuristic_name}\t{operator}\n"
        saved_solution = copy.deepcopy(env.current_solution)
        saved_state = copy.deepcopy(env.state_data)
        saved_algorithm_data = copy.deepcopy(env.algorithm_data)
        saved_key_value = env.key_value
        saved_is_complete = env.is_complete_solution

        best_score = None
        best_results = []
        output_file = open(os.path.join(output_dir, f"round_{round_index}.txt"), "w")
        output_file.write(f"selected_previous_heuristics\toperators\n{selected_previous_heuristics_str}")
        output_file.write("---------------\n")
        output_file.write(f"current_solution: \n{saved_solution}\n")
        output_file.write(f"is_complete_solution\t{saved_is_complete}\n")
        if saved_is_complete:
            output_file.write(f"key_value\t{saved_key_value}\n")

        output_file.write("---------------\n")
        output_file.write(f"heuristic\t{score_calculation.__name__}\tresults\n")
        for heuristic_file in os.listdir(heuristic_dir):
            env.reset()
            heuristic_name = heuristic_file.split(".")[0]
            results = []
            operators = []
            
            for search_index in range(search_time):
                env.current_solution = saved_solution
                env.state_data = saved_state
                env.algorithm_data = saved_algorithm_data
                operator = env.run_heuristic(load_heuristic(heuristic_file, heuristic_dir))
                operators.append(operator)
                if operator:
                    random_hh.run(env)
                    results.append(env.key_value)
                if (search_index + 1) % prune_ratio == 0:
                    if best_score and env.compare(best_score * prune_ratio, score_calculation(results)) > 0:
                        break
            if results == []:
                output_file.write(f"{heuristic_name}\tNone\tNone\n")
                continue
            score = score_calculation(results)
            results_str = ",".join([str(result) for result in sorted(results)])
            output_file.write(f"{heuristic_name}\t{score}\t{results_str}\n")

            if not best_score or env.compare(score, best_score) > 0:
                best_score = score
                best_heuristics = heuristic_name
                next_operator = next((operator for operator in operators if operator is not False), None)
                best_operators_str = [str(operator) for operator in operators]
                best_results = results
        selected_previous_heuristics.append(best_heuristics)
        previous_operator.append(next_operator)
        output_file.write("---------------\n")
        if saved_is_complete and env.is_complete_solution:
            if len(best_results) == 0 or (env.compare(saved_key_value, min(best_results)) >= 0 and env.compare(saved_key_value, max(best_results)) >= 0):
                output_file.write(f"Stop\n")
                output_file.close()
                break
        output_file.write(f"best_heuristics\t{best_heuristics}\n")
        output_file.write(f"best_operators\t{best_operators_str}\n")
        output_file.write(f"next_operator\t{next_operator}\n")
        output_file.close()

    output_file = open(os.path.join(output_dir, f"finished.txt"), "w")
    output_file.write(f"Total rounds: {round_index}")
    output_file.close()

def main():
    args = parse_arguments()
    problem = args.problem
    data_path = args.data_path
    heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics") if args.heuristic_dir is None else args.heuristic_dir
    search_time = args.search_time
    score_calculation = eval(args.score_calculation)
    prune_frequency = args.prune_frequency
    prune_ratio = args.prune_ratio
    output_dir = args.output_dir

    base_data_dir = os.getenv("AMLT_DATA_DIR") if os.getenv("AMLT_DATA_DIR") else "output"
    data_path = os.path.join(base_data_dir, data_path)

    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    os.makedirs(base_output_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = data_path.split(os.sep)[-1]
    output_dir = os.path.join(base_output_dir, problem, "heuristic_selection_data_collection", output_dir, f"{data_name}.{datetime_str}.result")

    data_collection(
        problem=problem,
        data_path=data_path,
        heuristic_dir=heuristic_dir,
        score_calculation=score_calculation,
        prune_frequency=prune_frequency,
        prune_ratio=prune_ratio,
        search_time=search_time,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()    
