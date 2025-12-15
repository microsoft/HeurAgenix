import os
import traceback
import math
from src.problems.base.env import BaseEnv
from src.util.function_to_tool import convert_function_to_tool
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.tts_bon import tts_bon
from src.util.util import find_closest_match, load_function, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        heuristic_pool: list[str],
        problem: str,
        tool_calling: bool=False,
        iterations_scale_factor: float=2.0,
        selection_frequency: int=5,
        num_candidate_heuristics: int=3,
        rollout_budget: int=10,
        problem_state_content_threshold: int=1000,
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.heuristic_pool = [heuristic.split(".")[0] for heuristic in heuristic_pool]
        self.tool_calling = tool_calling
        self.iterations_scale_factor = iterations_scale_factor
        self.selection_frequency = selection_frequency
        self.num_candidate_heuristics = num_candidate_heuristics
        self.rollout_budget = rollout_budget
        self.problem_state_content_threshold = problem_state_content_threshold

        self.heuristic_functions = {}
        self.heuristic_names = {}
        self.heuristic_pool_doc = ""
        heuristic_id = "A"
        for heuristic in heuristic_pool:
            heuristic_name = heuristic.split(".")[0]
            heuristic_code = open(search_file(heuristic_name + ".py", problem), "r", encoding="utf-8").read()

            self.heuristic_functions[heuristic_id] = load_function(heuristic, problem=problem)
            self.heuristic_names[heuristic_id] = heuristic_name
            self.heuristic_pool_doc += heuristic_id + "," + extract_function_with_short_docstring(heuristic_code, heuristic.split(".")[0]).split("def ")[-1] + "\n"
            heuristic_id = chr(ord(heuristic_id) + 1)
        self.last_heuristic_id = chr(ord(heuristic_id) - 1)

        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")
        self.get_observation_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_observation_problem_state")

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        max_rounds = math.ceil(max_steps / self.selection_frequency)
        selection_round = 0
        hidden_heuristics = []
        heuristic_traject = []

        # Load background
        # prompt_dict = self.llm_client.load_background(self.problem, background_file="background_without_code.txt")
        # Load system prompt 
        system_prompt_file = os.path.join("src", "problems", "base", "prompt", "system_prompt.txt")
        system_prompt = open(system_prompt_file, encoding="UTF-8").read()
        self.llm_client.messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        prompt_dict = {}
        prompt_dict["problem"] = self.problem
        prompt_dict["problem_description"] = open(search_file("problem_description.txt", self.problem), encoding="utf-8").read()

        # Generate global heuristic value
        instance_data = env.instance_data
        instance_problem_state = self.get_instance_problem_state(instance_data)
        prompt_dict["instance_problem_state"] = filter_dict_to_str([instance_data, instance_problem_state], self.problem_state_content_threshold)

        next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)
        while selection_round <= max_rounds and env.continue_run:
            try:
                if env.is_complete_solution:
                    env.dump_result()
                
                self.llm_client.messages = []

                # Load heuristic pool
                prompt_dict["heuristic_pool_introduction"] = self.heuristic_pool_doc

                # Generate state heuristic value
                solution_data = {"current_solution": env.current_solution, env.key_item: env.key_value}
                solution_problem_state = next_solution_problem_state
                prompt_dict["solution_problem_state"] = filter_dict_to_str([solution_data, solution_problem_state], self.problem_state_content_threshold)

                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                else:
                    heuristic_trajectory_str = "\n".join([f"-----\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for items in heuristic_traject[-5:]])
                prompt_dict["discuss_round"] = str(selection_round)
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["max_steps"] = max_steps
                prompt_dict["selection_frequency"] = self.selection_frequency
                prompt_dict["max_rounds"] = max_rounds
                prompt_dict["num_candidate_heuristics"] = self.num_candidate_heuristics
                prompt_dict["demo_heuristic_str"] = f"A/B/.../{self.last_heuristic_id}"
                
                self.llm_client.load("heuristic_selection", prompt_dict)
                response = self.llm_client.chat()
                self.llm_client.dump(f"step_{selection_round}")
                selected_heuristic_id = extract(response, key="Selected heuristic id")
                selected_heuristic_name = self.heuristic_names[selected_heuristic_id]
                selected_heuristic = self.heuristic_functions[selected_heuristic_id]
                # Record selection and observation
                pre_observation = self.get_observation_problem_state(solution_problem_state)
                pre_observation[env.key_item] = env.key_value
                for _ in range(self.selection_frequency):
                    env.run_heuristic(selected_heuristic, add_record_item={"step": selection_round})
                next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)
                next_observation = self.get_observation_problem_state(next_solution_problem_state)
                next_observation[env.key_item] = env.key_value
                heuristic_dict = {
                    "Selection Index": selection_round,
                    "Selected heuristic ID": selected_heuristic_id,
                    "Heuristic": selected_heuristic_name,
                }
                for key in pre_observation.keys():
                    heuristic_dict["Delta of " + key] = f"From {pre_observation[key]} to {next_observation[key]}"
                heuristic_traject.append(heuristic_dict)
                selection_round += 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
