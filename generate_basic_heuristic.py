import argparse
import os
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.llm_client.get_llm_client import get_llm_client


def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Specifies the type of combinatorial optimization problem.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Optional flag to conduct a preliminary smoke test.")
    parser.add_argument("-l", "--llm_config_file", type=str, default=os.path.join("output", "llm_config", "azure_gpt_4o.json"), help="Path to the language model configuration file. Default is azure_gpt_4o.json.")
    parser.add_argument("-s", "--source", choices=["llm", "paper", "related_problem"], default="llm", help="Source for generating heuristics: 'llm' for automated model generation, 'paper' for literature-based methods, 'related_problem' for adaptations from similar problems.")
    parser.add_argument("-pp", "--paper_path", type=str, help="Specify the path to a LaTeX paper file or directory containing heuristic descriptions.")
    parser.add_argument("-r", "--related_problems", type=str, default="all", help="Comma-separated list of related problem domains to draw heuristic insights from.")
    parser.add_argument("-d", "--reference_data", type=str, default=None, help="Optional path to reference datasets, used when generating heuristics tailored to specific data distributions.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    problem = args.problem
    source = args.source
    smoke_test = args.smoke_test
    llm_config_file = args.llm_config_file

    prompt_dir=os.path.join("src", "problems", "base", "prompt")
    output_dir=os.path.join("output", problem, "generate_heuristic")
    llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

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