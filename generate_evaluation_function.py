import argparse
import os
from src.pipeline.evaluation_function_generator import EvaluationFunctionGenerator
from src.util.llm_client.get_llm_client import get_llm_client


def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")
    parser.add_argument("-l", "--llm_config_file", type=str, default="AzureGPT", help="LLM config file to use.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    smoke_test = args.smoke_test
    llm_config_file = args.llm_config_file

    prompt_dir=os.path.join("src", "problems", "base", "prompt")
    output_dir=output_dir=os.path.join("output", problem, "generate_evaluation_function")
    llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

    evaluation_function_generator = EvaluationFunctionGenerator(llm_client=llm_client, problem=problem)
    evaluation_function_generator.generate_evaluation_function(smoke_test=smoke_test)

if __name__ == "__main__":
    main()
