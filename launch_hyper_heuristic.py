import argparse
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.gpt_selection import GPTSelectionHyperHeuristic
from src.pipeline.hyper_heuristics.gpt_deep_selection import GPTDeepSelectionHyperHeuristic
from src.util.gpt_helper import GPTHelper

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-e", "--heuristic", type=str, required=True, help="Name or path of the heuristic function. Use 'gpt_hh' / 'gpt_deep_hh' /'random_hh' for GPT/random selection from the heuristic directory, and 'or_solver' for OR result.")
    parser.add_argument("-d", "--heuristic_type", type=str, default="basic_heuristics", help="Directory containing heuristic functions.")
    parser.add_argument("-si", "--search_interval", type=int, default=None, help="Search interval for deep hh mode.")
    parser.add_argument("-st", "--search_time", type=int, default=None, help="Search time for deep hh mode.")
    parser.add_argument("-c", "--test_case", type=str, default=None, help="Data name for single test case.")
    parser.add_argument("-t", "--test_dir", type=str, default=None, help="Directory for the whole test set.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    heuristic = args.heuristic
    heuristic_type = args.heuristic_type
    test_case = args.test_case
    search_interval = args.search_interval
    search_time = args.search_time
    if test_case is None:
        test_dir = os.path.join("output", problem, "data", "test_data") if args.test_dir is None else args.test_dir
    test_cases = [os.path.join(test_dir, test_case) for test_case in os.listdir(test_dir)] if test_case is None else [test_case]
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    heuristic = heuristic.split(os.sep)[-1].split(".")[0]

    heuristic_pool = os.listdir(os.path.join("src", "problems", problem, "heuristics", heuristic_type))
    evolved_names = [heuristic[:-8] for heuristic in heuristic_pool]
    heuristic_pool += [file for file in os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")) if file[:-8] not in evolved_names]

    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")

    gpt_helper = GPTHelper(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
    if heuristic == "gpt_hh":
        output_dir = f"{heuristic}.{datetime_str}"
        output_dir = f"{heuristic}.{heuristic_type}.{datetime_str}"
        hyper_heuristic = GPTSelectionHyperHeuristic(gpt_helper=gpt_helper, heuristic_pool=heuristic_pool, problem=problem)
    elif heuristic == "gpt_deep_hh":
        output_dir = f"{heuristic}.{datetime_str}"
        output_dir = f"{heuristic}.{heuristic_type}.{datetime_str}"
        hyper_heuristic = GPTDeepSelectionHyperHeuristic(
            gpt_helper=gpt_helper,
            heuristic_pool=heuristic_pool,
            problem=problem,
            search_interval=search_interval,
            search_time=search_time,
        )
    elif heuristic == "random_hh":
        output_dir = f"{heuristic}.{datetime_str}"
        if heuristic_type == "evolved":
            output_dir = f"{heuristic}.{heuristic_type}.{datetime_str}"
        hyper_heuristic = RandomHyperHeuristic(heuristic_pool=heuristic_pool, problem=problem)
    elif heuristic == "or_solver":
        output_dir = "or_solver"
        module = importlib.import_module(f"src.problems.{problem}.or_solver")
        globals()["ORSolver"] = getattr(module, "ORSolver")
        hyper_heuristic = ORSolver(problem=problem)
    else:
        output_dir = heuristic
        hyper_heuristic = SingleHyperHeuristic(heuristic, problem=problem)

    for test_case in test_cases:
        env = Env(data_name=test_case)
        env.reset(output_dir)
        gpt_helper.reset(env.output_dir)
        validation_result = hyper_heuristic.run(env)
        if validation_result:
            env.dump_result()
            print(os.path.join(env.output_dir, "result.txt"), heuristic, test_case, env.key_item, env.key_value)
        else:
            print("Invalid solution", heuristic, test_case)


if __name__ == "__main__":
    main()
