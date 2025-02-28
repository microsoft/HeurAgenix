import argparse
import os
import os
# 设置环境变量，强制修改缓存目录
os.environ["TRANSFORMERS_CACHE"] = "/Data/haolong/model_deploy/models_cache/"
from src.pipeline.heuristic_evolver import HeuristicEvolver
from src.util.gpt_helper import GPTHelper
from src.util.util import search_file

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Evolve heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-be", "--basic_heuristic", type=str, required=True, help="Name or path of the basic heuristic.")
    parser.add_argument("-t", "--train_dir", type=str, default="train_data", help="Directory for the training dataset.")
    parser.add_argument("-v", "--validation_dir", type=str, default="validation_data", help="Directory for the validation dataset.")
    parser.add_argument("-pe", "--perturbation_heuristic", type=str, default=None, help="Name or path of the perturbation heuristic.")
    parser.add_argument("-pr", "--perturbation_ratio", type=float, default=0.1, help="Ratio of perturbation.")
    parser.add_argument("-pt", "--perturbation_time", type=float, default=1000, help="Maximum number of perturbation times.")
    parser.add_argument("-f", "--filter_num", type=int, default=1, help="Number of heuristics to keep after each validation.")
    parser.add_argument("-r", "--evolution_rounds", type=int, default=3, help="Number of evolution rounds.")
    parser.add_argument("-l", "--time_limit", type=int, default=10, help="Time limit for running.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    basic_heuristic_file = args.basic_heuristic
    perturbation_heuristic_file = args.perturbation_heuristic
    perturbation_ratio = args.perturbation_ratio
    perturbation_time = args.perturbation_time
    filter_num = args.filter_num
    evolution_rounds = args.evolution_rounds
    time_limitation = args.time_limit
    smoke_test = args.smoke_test

    train_dir = search_file(args.train_dir, problem)
    validation_dir = search_file(args.validation_dir, problem)
    if args.validation_dir is None:
        validation_dir = args.train_dir
    if not basic_heuristic_file.endswith(".py"):
        basic_heuristic_file += ".py"
    if not perturbation_heuristic_file.endswith(".py"):
        perturbation_heuristic_file += ".py"

    gpt_helper = GPTHelper(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
    gpt_evolver = HeuristicEvolver(gpt_helper, problem, train_dir, validation_dir)
    evolved_heuristics = gpt_evolver.evolution(
        basic_heuristic_file,
        perturbation_heuristic_file,
        perturbation_ratio=perturbation_ratio,
        perturbation_time=perturbation_time,
        filtered_num=filter_num,
        evolution_round=evolution_rounds,
        time_limitation=time_limitation,
        smoke_test=smoke_test
    )
    print(evolved_heuristics)

if __name__ == "__main__":
    main()
