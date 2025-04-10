import argparse
import os
import re
from src.pipeline.heuristic_evolver_v2 import HeuristicEvolver
from src.util.util import search_file

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Evolve heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-e", "--basic_heuristic", type=str, required=True, help="Name or path of the basic heuristic.")
    parser.add_argument("-t", "--train_dir", type=str, default="train_data", help="Directory for the training dataset.")
    parser.add_argument("-v", "--validation_dir", type=str, default="validation_data", help="Directory for the validation dataset.")
    parser.add_argument("-pe", "--perturbation_heuristic", type=str, default=None, help="Name or path of the perturbation heuristic.")
    parser.add_argument("-pr", "--perturbation_ratio", type=float, default=0.1, help="Ratio of perturbation.")
    parser.add_argument("-pt", "--perturbation_time", type=float, default=1000, help="Maximum number of perturbation times.")
    parser.add_argument("-i", "--max_refinement_round", type=int, default=5, help="Number of refinement rounds.")
    parser.add_argument("-f", "--filter_num", type=int, default=1, help="Number of heuristics to keep after each validation.")
    parser.add_argument("-r", "--evolution_rounds", type=int, default=3, help="Number of evolution rounds.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")
    parser.add_argument("-l", "--llm_type", type=str, default="AzureGPT", choices=["AzureGPT", "APIModel"], help="LLM Type to use.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    basic_heuristic_file = args.basic_heuristic
    perturbation_heuristic_file = args.perturbation_heuristic
    perturbation_ratio = args.perturbation_ratio
    perturbation_time = args.perturbation_time
    max_refinement_round = args.max_refinement_round
    filter_num = args.filter_num
    evolution_rounds = args.evolution_rounds
    smoke_test = args.smoke_test
    llm_type= args.llm_type

    train_dir = search_file(args.train_dir, problem)
    validation_dir = search_file(args.validation_dir, problem)
    if args.validation_dir is None:
        validation_dir = args.train_dir
    if not basic_heuristic_file.endswith(".py"):
        basic_heuristic_file += ".py"
    basic_heuristic_file = search_file(basic_heuristic_file, problem)
    if perturbation_heuristic_file is None:
        perturbation_heuristic_file = [file_name for file_name in os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")) if re.match( r"random_....\.py", file_name)]
        if perturbation_heuristic_file == []:
            raise Exception("No perturbation heuristics")
        perturbation_heuristic_file = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics", perturbation_heuristic_file[0])
    else:
        if not perturbation_heuristic_file.endswith(".py"):
            perturbation_heuristic_file += ".py"
        perturbation_heuristic_file = search_file(perturbation_heuristic_file, problem)

    if llm_type == "AzureGPT":
        from src.util.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
    elif llm_type == "APIModel":
        from src.util.api_model_client import APIModelClient
        llm_client = APIModelClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
    heuristic_evolver = HeuristicEvolver(llm_client, problem, train_dir, validation_dir)
    evolved_heuristics = heuristic_evolver.evolution(
        basic_heuristic_file,
        perturbation_heuristic_file,
        perturbation_ratio=perturbation_ratio,
        perturbation_time=perturbation_time,
        max_refinement_round=max_refinement_round,
        filtered_num=filter_num,
        evolution_round=evolution_rounds,
        smoke_test=smoke_test
    )
    print(evolved_heuristics)

if __name__ == "__main__":
    main()
