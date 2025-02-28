import os
# 设置环境变量，强制修改缓存目录
os.environ["TRANSFORMERS_CACHE"] = "/Data/haolong/model_deploy/models_cache/"

import argparse
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.single_construct_single_improve import SingleConstructiveSingleImproveHyperHeuristic
from src.pipeline.hyper_heuristics.gpt_selection import GPTSelectionHyperHeuristic
from src.util.gpt_helper import GPTHelper
import json

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-e", "--heuristic", type=str, required=True, help="Name or path of the heuristic function. Use 'gpt_hh'/'random_hh' for GPT/random selection from the heuristic directory, and 'or_solver' for OR result.")
    parser.add_argument("-d", "--heuristic_dir", type=str, default=None, help="Directory containing heuristic functions.")
    parser.add_argument("-c", "--test_case", type=str, default=None, help="Path for single test case.")
    parser.add_argument("-t", "--test_dir", type=str, default=None, help="Directory for the whole test set.")
    parser.add_argument("-r", "--dump_trajectory", action='store_true', help="Whether to dump trajectory.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Output experiment name.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    heuristic = args.heuristic
    heuristic_dir = args.heuristic_dir
    test_case = args.test_case
    if test_case is None:
        test_dir = [os.path.join(args.test_dir,i) for i in os.listdir(args.test_dir)]
    else:
        test_dir = [test_case]
    # test_dir = args.test_dir if test_case is None else [test_case] 
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpt_setting = json.load(open("gpt_setting.json"))

    api_type = gpt_setting.get("api_type", "azure")

    model = gpt_setting["model"]

    local_model = gpt_setting["local_model"]
    local_model_name = local_model.split('/')[-1]
    if api_type == "azure":
        output_dir = args.output_dir if args.output_dir is not None else f"{heuristic}.{model}.{datetime_str}"
    else:
        output_dir = args.output_dir if args.output_dir is not None else f"{heuristic}.{local_model_name}.{datetime_str}"
    if heuristic_dir is None:
        heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")

    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")

    for test_case in test_dir:
        env = Env(data_name=test_case)
        env.reset(output_dir)

        if heuristic == "gpt_hh":
            gpt_helper = GPTHelper(
                prompt_dir=os.path.join("src", "problems", "base", "prompt"),
                output_dir=env.output_dir,
            )
            hyper_heuristic = GPTSelectionHyperHeuristic(gpt_helper=gpt_helper, heuristic_dir=heuristic_dir, problem=problem)
        elif heuristic == "random_hh":
            hyper_heuristic = RandomHyperHeuristic(heuristic_dir=heuristic_dir, problem=problem)
        elif heuristic == "or_solver":
            module = importlib.import_module(f"src.problems.{problem}.or_solver")
            globals()["ORSolver"] = getattr(module, "ORSolver")
            hyper_heuristic = ORSolver(problem=problem)
        else:
            hyper_heuristic = SingleHyperHeuristic(heuristic, problem=problem)

        validation_result = hyper_heuristic.run(env)
        if validation_result:
            env.dump_result(args.dump_trajectory)
            print(os.path.join(env.output_dir, "result.txt"), env.key_item, env.key_value)


if __name__ == "__main__":
    main()
