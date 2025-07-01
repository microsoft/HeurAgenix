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
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Specifies the type of combinatorial optimization problem.")
    parser.add_argument("-e", "--heuristic", type=str, required=True, help=": Specifies which heuristic function or strategy to apply. 'heuristic_function_name': Directly specify a heuristic function. 'llm_hh': Utilizes LLM for rapid heuristic selection from the directory. 'random_hh': Randomly selects a heuristic from the directory. 'or_solver': Uses an exact OR solver, where applicable.")
    parser.add_argument("-l", "--llm_config_file", type=str, default=os.path.join("output", "llm_config", "azure_gpt_4o.json"), help="Path to the language model configuration file. Default is azure_gpt_4o.json.")
    parser.add_argument("-d", "--heuristic_dir", type=str, default="basic_heuristics", help="Directory containing heuristics for llm_hh or random_hh. Default is 'basic_heuristics'.")
    parser.add_argument("-t", "--test_data", type=str, default="test_data", help="Path to a specific test data file. Defaults to testing all files in the `test_data` directory if not specified.")
    parser.add_argument("-n", "--iterations_scale_factor", type=float, default=2.0, help="Scale factor determining total heuristic steps relative to problem size. Default is 2.0.")
    parser.add_argument("-m", "--steps_per_selection", type=int, default=5, help="Number of steps executed per heuristic selection in LLM mode. Default is 5.")
    parser.add_argument("-c", "--num_candidate_heuristics", type=int, default=1, help="Number of candidate heuristics considered in LLM mode. 1 represents select by LLM without TTS. Default is 1.")
    parser.add_argument("-b", "--rollout_budget", type=int, default=0, help="Number of Monte-Carlo evaluations per heuristic in LLM mode. 0 represents select by LLM without TTS. Default is 0.")
    parser.add_argument("-r", "--result_dir", type=str, default="result", help="Target directory for saving results. Default is 'result'.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    heuristic = args.heuristic
    heuristic_dir = args.heuristic_dir
    test_data = args.test_data
    llm_config_file = args.llm_config_file
    iterations_scale_factor = args.iterations_scale_factor
    steps_per_selection = args.steps_per_selection
    num_candidate_heuristics = args.num_candidate_heuristics
    rollout_budget = args.rollout_budget
    result_dir = args.result_dir

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    heuristic = heuristic.split(os.sep)[-1].split(".")[0]
    heuristic_pool = os.listdir(os.path.join("src", "problems", problem, "heuristics", heuristic_dir))

    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"

    if heuristic == "llm_hh":
        prompt_dir = os.path.join("src", "problems", "base", "prompt")
        llm_client = get_llm_client(llm_config_file, prompt_dir, None)
        llm_name = llm_config_file.split(os.sep)[-1].split(".")[0]
        experiment_name = f"{heuristic}.{heuristic_dir}.{llm_name}.n{iterations_scale_factor}m{steps_per_selection}c{num_candidate_heuristics}b{rollout_budget}.{datetime_str}"
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
        experiment_name = f"{heuristic}.{heuristic_dir}.{datetime_str}"
        hyper_heuristic = RandomHyperHeuristic(heuristic_pool=heuristic_pool, problem=problem, iterations_scale_factor=iterations_scale_factor)
    elif heuristic == "or_solver":
        experiment_name = "or_solver"
        module = importlib.import_module(f"src.problems.{problem}.or_solver")
        globals()["ORSolver"] = getattr(module, "ORSolver")
        hyper_heuristic = ORSolver(problem=problem)
    else:
        experiment_name = heuristic
        hyper_heuristic = SingleHyperHeuristic(heuristic=heuristic, problem=problem)

    module = importlib.import_module(f"src.problems.{problem}.env")
    globals()["Env"] = getattr(module, "Env")

    if test_data == "test_data":
        test_data = os.listdir(search_file("test_data", problem))
    else:
        test_data = [test_data]

    for data_name in test_data:
        env = Env(data_name=data_name)
        output_dir = os.path.join(base_output_dir, problem, result_dir, env.data_ref_name, experiment_name)
        env.reset(output_dir)

        paras = '\n'.join(f'{key}={value}' for key, value in vars(args).items()) 
        paras += f"\ndata_path={env.data_path}"
        with open(os.path.join(env.output_dir, "parameters.txt"), 'w') as file:
            file.write(paras)

        if heuristic == "llm_hh":
            llm_client.reset(env.output_dir)
        validation_result = hyper_heuristic.run(env)
        if validation_result:
            env.dump_result()
            print(os.path.join(env.output_dir, "result.txt"), heuristic, data_name, env.key_item, env.key_value)
        else:
            print("Invalid solution", heuristic, data_name)


if __name__ == "__main__":
    main()
