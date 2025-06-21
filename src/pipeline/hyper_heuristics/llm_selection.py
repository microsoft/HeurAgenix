import traceback
from src.problems.base.env import BaseEnv
from src.util.util import find_closest_match, load_function, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.tts_bon import tts_bon


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float=2.0,
        steps_per_selection: int=5,
        num_candidate_heuristics: int=3,
        rollout_budget: int=10,
        problem_state_content_threshold: int=1000,
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.heuristic_pool = [heuristic.split(".")[0] for heuristic in heuristic_pool]
        self.iterations_scale_factor = iterations_scale_factor
        self.steps_per_selection = steps_per_selection
        self.num_candidate_heuristics = num_candidate_heuristics
        self.rollout_budget = rollout_budget
        self.problem_state_content_threshold = problem_state_content_threshold

        self.heuristic_docs = {
            heuristic: extract_function_with_short_docstring(open(search_file(heuristic + ".py", problem)).read(), heuristic) 
            for heuristic in self.heuristic_pool}
        self.heuristic_functions = {
            heuristic.split(".")[0]: load_function(heuristic, problem=self.problem)
            for heuristic in self.heuristic_pool}
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")
        self.get_observation_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_observation_problem_state")

    def run(self, env:BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        selection_round = 0
        hidden_heuristics = []
        heuristic_traject = []

        # Load background
        prompt_dict = self.llm_client.load_background(self.problem, background_file="background_without_code.txt")

        # Generate global heuristic value
        instance_data = env.instance_data
        instance_problem_state = self.get_instance_problem_state(instance_data)
        prompt_dict["instance_problem_state"] = filter_dict_to_str([instance_data, instance_problem_state], self.problem_state_content_threshold)

        next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution, env.get_key_value)
        while selection_round * self.steps_per_selection <= max_steps and env.continue_run:
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
                prompt_dict["selection_frequency"] = self.steps_per_selection
                prompt_dict["num_candidate_heuristics"] = self.num_candidate_heuristics
                prompt_dict["demo_heuristic_str"] = ",".join([f"heuristic_name_{i + 1}"for i in range(self.num_candidate_heuristics)])
                
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
                    self.steps_per_selection,
                    self.rollout_budget,
                )
                # Record selection and observation
                pre_observation = self.get_observation_problem_state(solution_problem_state)
                pre_observation[env.key_item] = env.key_value
                for _ in range(self.steps_per_selection):
                    env.run_heuristic(self.heuristic_functions[selected_heuristic_name])
                next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution, env.get_key_value)
                next_observation = self.get_observation_problem_state(next_solution_problem_state)
                next_observation[env.key_item] = env.key_value
                heuristic_dict = {
                    "Selection Index": selection_round,
                    "Heuristic": selected_heuristic_name,
                }
                for key in pre_observation.keys():
                    heuristic_dict["Delta of " + key] = next_observation[key] - pre_observation[key]
                heuristic_traject.append(heuristic_dict)
                selection_round += 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
