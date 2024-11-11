import argparse
import os
from src.pipeline.heuristic_evolver import HeuristicEvolver
from src.util.gpt_helper import GPTHelper

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Evolve heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-b", "--basic_heuristic", type=str, required=True, help="Name or path of the basic heuristic.")
    parser.add_argument("-t", "--train_dir", type=str, default=None, help="Directory for the training dataset.")
    parser.add_argument("-v", "--validation_dir", type=str, default=None, help="Directory for the validation dataset.")
    parser.add_argument("-ph", "--perturb_heuristic", type=str, default=None, help="Name or path of the perturbation heuristic.")
    parser.add_argument("-pr", "--perturb_ratio", type=float, default=0.1, help="Ratio of perturbation.")
    parser.add_argument("-pt", "--perturb_time", type=float, default=1000, help="Maximum number of perturbation times.")
    parser.add_argument("-f", "--filter_num", type=int, default=1, help="Number of heuristics to keep after each validation.")
    parser.add_argument("-r", "--evolution_rounds", type=int, default=3, help="Number of evolution rounds.")
    parser.add_argument("-l", "--time_limit", type=int, default=10, help="Time limit for running.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    perturb_ratio = args.perturb_ratio
    perturb_time = args.perturb_time
    filter_num = args.filter_num
    evolution_rounds = args.evolution_rounds
    time_limitation = args.time_limit
    smoke_test = args.smoke_test

    if os.path.exists(args.basic_heuristic):
        basic_heuristic_file = args.basic_heuristic
    else:
        basic_heuristic_file = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics", args.basic_heuristic + ".py")
    if args.perturb_heuristic is None:
        for file in os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")):
            if file.split("_")[0] == "random" and len(file.split("_")) == 2:
                perturb_heuristic_file = file
    else:
        if os.path.exists(args.perturb_heuristic):
            perturb_heuristic_file = args.perturb_heuristic
        else:
            perturb_heuristic_file = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics", args.perturb_heuristic + ".py")
    if args.train_dir is None:
        train_dir = os.path.join("src", "problems", problem, "data", "train_data")
    else:
        train_dir = args.train_dir
    if args.validation_dir is None:
        if os.path.exists(os.path.join("src", "problems", problem, "data", "validation_data")):
            validation_dir = os.path.join("src", "problems", problem, "data", "validation_data")
        else:
            validation_dir = os.path.join("src", "problems", problem, "data", "train_data")
    else:
        validation_dir = args.validation_dir

    gpt_helper = GPTHelper(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
    gpt_evolver = HeuristicEvolver(gpt_helper, problem, train_dir, validation_dir)
    evolved_heuristics = gpt_evolver.evolution(
        basic_heuristic_file,
        perturb_heuristic_file,
        perturb_ratio=perturb_ratio,
        perturb_time=perturb_time,
        filtered_num=filter_num,
        evolution_round=evolution_rounds,
        time_limitation=time_limitation,
        smoke_test=smoke_test
    )
    print(evolved_heuristics)

if __name__ == "__main__":
    main()
