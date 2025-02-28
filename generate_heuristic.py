import argparse
import os
import os
# 设置环境变量，强制修改缓存目录
os.environ["TRANSFORMERS_CACHE"] = "/Data/haolong/model_deploy/models_cache/"
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.gpt_helper import GPTHelper

def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-s", "--source", choices=["gpt", "paper", "related_problem"], required=True, help="Source for generating heuristics.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Run a smoke test.")
    parser.add_argument("-pp", "--paper_path", type=str, help="Path to Latex paper file or directory.")
    parser.add_argument("-r", "--related_problems", type=str, default="all", help="Comma-separated list of related problems to reference.")
    parser.add_argument("-d", "--reference_data", type=str, default=None, help="Path for reference data.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    problem = args.problem
    source = args.source
    smoke_test = args.smoke_test
    gpt_helper = GPTHelper(
        prompt_dir=os.path.join("src", "problems", "base", "prompt"),
        output_dir=os.path.join("output", problem, "generate_heuristic")
    )
    heuristic_generator = HeuristicGenerator(gpt_helper=gpt_helper, problem=problem)
    if source == "gpt":
        heuristic_generator.generate_from_gpt(reference_data=args.reference_data, smoke_test=smoke_test)
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