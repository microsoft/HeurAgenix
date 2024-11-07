import os
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.gpt_helper import GPTHelper

problem = "dposp"
smoke_test = True
related_problems = ["tsp", "cvrp", "jssp", "max_cut", "mkp"]
gpt_helper = GPTHelper(
    prompt_dir=os.path.join("src", "problems", "base", "prompt"),
    output_dir=os.path.join("output", problem, "generate_heuristic")
)
heuristic_generator = HeuristicGenerator(gpt_helper=gpt_helper, problem=problem)
heuristic_generator.generate_from_reference(related_problems=related_problems, smoke_test=smoke_test)