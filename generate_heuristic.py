import argparse
import os
from src.pipeline.heuristic_generator import HeuristicGenerator


def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-s", "--source", choices=["llm", "paper", "related_problem"], required=True, help="Source for generating heuristics.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")
    parser.add_argument("-pp", "--paper_path", type=str, help="Path to Latex paper file or directory.")
    parser.add_argument("-r", "--related_problems", type=str, default="all", help="Comma-separated list of related problems to reference.")
    parser.add_argument("-d", "--reference_data", type=str, default=None, help="Path for reference data.")
    parser.add_argument("-l", "--llm_type", type=str, default="AzureGPT", choices=["AzureGPT", "APIModel"], help="LLM Type to use.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    problem = args.problem
    source = args.source
    smoke_test = args.smoke_test
    llm_type= args.llm_type
    if llm_type == "AzureGPT":
        from src.util.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"), output_dir=os.path.join("output", problem, "generate_heuristic"))
    elif llm_type == "APIModel":
        from src.util.api_model_client import APIModelClient
        llm_client = APIModelClient(prompt_dir=os.path.join("src", "problems", "base", "prompt"), output_dir=os.path.join("output", problem, "generate_heuristic"))
    heuristic_generator = HeuristicGenerator(llm_client=llm_client, problem=problem)
    if source == "llm":
        heuristic_generator.generate_from_llm(reference_data=args.reference_data, smoke_test=smoke_test)
    elif source == "paper":
        heuristic_generator.generate_from_paper(paper_path=args.paper_path, reference_data=args.reference_data, smoke_test=smoke_test)
    elif source == "related_problem":
        if args.related_problems == "all":
            related_problems = [ref_problem for ref_problem in os.listdir(os.path.join("src", "problems")) if ref_problem not in ["base", problem]]
        else:
            related_problems = args.related_problems.split(",")
        heuristic_generator.generate_from_reference(related_problems=related_problems, reference_data=args.reference_data, smoke_test=smoke_test)

if __name__ == "__main__":
    main()