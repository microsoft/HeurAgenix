import argparse
import os
from src.pipeline.evaluation_function_generator import EvaluationFunctionGenerator

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")
    parser.add_argument("-l", "--llm_type", type=str, default="AzureGPT", choices=["AzureGPT", "APIModel"], help="LLM Type to use.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    smoke_test = args.smoke_test
    llm_type= args.llm_type
    if llm_type == "AzureGPT":
        from src.util.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"), output_dir=os.path.join("output", problem, "generate_evaluation_function"))
    elif llm_type == "APIModel":
        from src.util.api_model_client import APIModelClient
        llm_client = APIModelClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"), output_dir=os.path.join("output", problem, "generate_evaluation_function"))
    evaluation_function_generator = EvaluationFunctionGenerator(llm_client=llm_client, problem=problem)
    evaluation_function_generator.generate_evaluation_function(smoke_test=smoke_test)

if __name__ == "__main__":
    main()
