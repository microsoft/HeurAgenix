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

        self.heuristic_docs = {}
        self.heuristic_functions = {}
        self.tools = []
        for heuristic in self.heuristic_pool:
            heuristic_name = heuristic.split(".")[0]
            heuristic_code = open(search_file(heuristic_name + ".py", problem), "r", encoding="utf-8").read()
            self.heuristic_docs[heuristic_name] = extract_function_with_short_docstring(heuristic_code, heuristic) 
            self.heuristic_functions[heuristic_name] = load_function(heuristic, problem=self.problem)
            self.tools.append(convert_function_to_tool(heuristic_name, code=heuristic_code))

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
        prompt_dict = self.llm_client.load_background(self.problem, background_file="background_without_code.txt")

        # Generate global heuristic value
        instance_data = env.instance_data
        instance_problem_state = self.get_instance_problem_state(instance_data)
        prompt_dict["instance_problem_state"] = filter_dict_to_str([instance_data, instance_problem_state], self.problem_state_content_threshold)

        next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)
        while selection_round <= max_rounds and env.continue_run:
            try:
                if env.is_complete_solution:
                    env.dump_result()
                self.llm_client.load_chat("background")

                # Load heuristic pool
                heuristic_pool_doc = ""
                for heuristic in self.heuristic_pool:
                    if heuristic not in hidden_heuristics:
                        heuristic_pool_doc += self.heuristic_docs[heuristic] + "\n"
                prompt_dict["heuristic_pool_introduction"] = heuristic_pool_doc

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
                prompt_dict["demo_heuristic_str"] = ",".join([f"heuristic_name_{i + 1}"for i in range(self.num_candidate_heuristics)])
                
                if self.tool_calling:
                    self.llm_client.load("heuristic_selection_tool_calling", prompt_dict)
                    function_name_parameters = self.llm_client.chat_with_tools(self.tools)
                    self.llm_client.dump(f"step_{selection_round}")
                    candidate_heuristics = [function[0] for function in function_name_parameters]
                else:
                    self.llm_client.load("heuristic_selection", prompt_dict)
                    response = self.llm_client.chat()
                    self.llm_client.dump(f"step_{selection_round}")
                    candidate_heuristics = extract(response, key="Selected heuristic", sep=",")

                matched_candidate_heuristics = []
                for heuristic in candidate_heuristics:
                    matched_candidate_heuristic = find_closest_match(heuristic, self.heuristic_pool)
                    if matched_candidate_heuristic:
                        matched_candidate_heuristics.append(matched_candidate_heuristic)
                assert len(matched_candidate_heuristics) > 0
                
                # TTS selection
                selected_heuristic_name = tts_bon(
                    env,
                    matched_candidate_heuristics,
                    self.heuristic_pool,
                    self.problem,
                    self.iterations_scale_factor,
                    self.selection_frequency,
                    self.rollout_budget,
                )
                # Record selection and observation
                pre_observation = self.get_observation_problem_state(solution_problem_state)
                pre_observation[env.key_item] = env.key_value
                for _ in range(self.selection_frequency):
                    env.run_heuristic(self.heuristic_functions[selected_heuristic_name], add_record_item={"step": selection_round})
                next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)
                next_observation = self.get_observation_problem_state(next_solution_problem_state)
                next_observation[env.key_item] = env.key_value
                heuristic_dict = {
                    "Selection Index": selection_round,
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
