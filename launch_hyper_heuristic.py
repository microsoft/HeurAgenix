import argparse
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-e", "--heuristic", type=str, required=True, help="Name or path of the heuristic function. Use 'llm_hh' / 'llm_deep_hh' /'random_hh' for LLM/random selection from the heuristic directory, and 'or_solver' for OR result.")
    parser.add_argument("-d", "--heuristic_dir", type=str, default="basic_heuristics", help="Directory containing heuristic functions.")
    parser.add_argument("-t", "--test_case", type=str, default=None, help="Data or directory name for test case(s).")
    parser.add_argument("-l", "--llm_config_file", type=str, default=os.path.join("output", "llm_config", "azure_gpt_4o.json"), help="LLM config file in llm_hh.")
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0, help="Scale factor for determining total heuristic steps based on problem size")
    parser.add_argument("-m", "--steps_per_selection", type=int, default=5, help="Number of steps each heuristic selection should execute in llm_hh mode.")
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1, help="Number of candidate heuristics from llm in llm_hh mode.")
    parser.add_argument("-b", "--rollout_budget", type=int, default=0, help="Number of Monte-Carlo evaluation for each heuristic in llm_hh mode.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    heuristic = args.heuristic
    heuristic_dir = args.heuristic_dir
    test_case = args.test_case
    llm_config_file = args.llm_config_file
    iterations_scale_factor = args.iterations_scale_factor
    steps_per_selection = args.steps_per_selection
    num_candidate_heuristics = args.num_candidate_heuristics
    rollout_budget = args.rollout_budget

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    heuristic = heuristic.split(os.sep)[-1].split(".")[0]
    heuristic_pool = os.listdir(os.path.join("src", "problems", problem, "heuristics", heuristic_dir))

    if heuristic == "llm_hh":
        prompt_dir = os.path.join("src", "problems", "base", "prompt")
        llm_client = get_llm_client(llm_config_file, prompt_dir, None)
        llm_name = llm_config_file.split(os.sep)[-1].split(".")[0]
        output_dir = f"{heuristic}.{heuristic_dir}.{llm_name}.{datetime_str}"
        hyper_heuristic = LLMSelectionHyperHeuristic(
            llm_client=llm_client,
            heuristic_pool=heuristic_pool,
            problem=problem,
            iterations_scale_factor=iterations_scale_factor,
            steps_per_selection=steps_per_selection,
            num_candidate_heuristics=num_candidate_heuristics,
            rollout_budget=rollout_budget,
        )
    elif heuristic == "random_hh":
        output_dir = f"{heuristic}.{heuristic_dir}.{datetime_str}"
        hyper_heuristic = RandomHyperHeuristic(heuristic_pool=heuristic_pool, problem=problem, iterations_scale_factor=iterations_scale_factor)
    elif heuristic == "or_solver":
        output_dir = "or_solver"
        module = importlib.import_module(f"src.problems.{problem}.or_solver")
        globals()["ORSolver"] = getattr(module, "ORSolver")
        hyper_heuristic = ORSolver(problem=problem)
    else:
        output_dir = heuristic
        hyper_heuristic = SingleHyperHeuristic(heuristic=heuristic, problem=problem)

    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")

    if test_case is None:
        test_case = os.path.join("output", problem, "data", "test_data")
    
    test_case = search_file(test_case, problem)
    if os.path.isdir(test_case):
        test_case = os.listdir(test_case)
    elif os.path.isfile(test_case):
        test_case = [test_case]
    else:
        raise ValueError(f"Invalid test case: {test_case}")
    
    for data_name in test_case:
        env = Env(data_name=data_name)
        env.reset(output_dir)

        paras = '\n'.join(f'{key}={value}' for key, value in vars(args).items()) 
        paras += f"\ndata_path={env.data_path}"
        with open(os.path.join(env.output_dir, "para.txt"), 'w') as file:
            file.write(paras)

        if heuristic == "llm_hh":
            llm_client.reset(env.output_dir)
        validation_result = hyper_heuristic.run(env)
        if validation_result:
            env.dump_result()
            print(os.path.join(env.output_dir, "result.txt"), heuristic, data_name, env.key_item, env.key_value)
        else:
            print("Invalid solution", heuristic, test_case)


if __name__ == "__main__":
    main()
