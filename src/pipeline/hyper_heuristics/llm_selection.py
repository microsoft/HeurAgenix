import os
import traceback
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.tts_bon import tts_bon


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        heuristic_pool: list[str],
        problem: str,
        selection_frequency: int=5,
        num_candidate_heuristics: int=3,
        rollout_budget: int=10,
        data_feature_content_threshold: int=1000,
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.heuristic_pool = [heuristic.split(".")[0] for heuristic in heuristic_pool]
        self.selection_frequency = selection_frequency
        self.num_candidate_heuristics = num_candidate_heuristics
        self.rollout_budget = rollout_budget
        self.data_feature_content_threshold = data_feature_content_threshold

        self.heuristic_docs = {
            heuristic: extract_function_with_short_docstring(open(search_file(heuristic + ".py", problem)).read(), heuristic) 
            for heuristic in self.heuristic_pool}
        self.heuristic_functions = {
            heuristic.split(".")[0]: load_heuristic(heuristic, problem=self.problem)
            for heuristic in self.heuristic_pool}
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")

    def run(self, env:BaseEnv, max_steps: int=None, data_feature_content_threshold: int=1000, **kwargs) -> bool:
        selection_round = 0
        max_steps = max_steps if max_steps is not None else env.construction_steps * 3
        hidden_heuristics = []
        heuristic_traject = []

        # Load background
        prompt_dict = self.llm_client.load_background(self.problem)

        # Generate global heuristic value
        global_data = env.global_data
        global_data_feature = self.get_global_data_feature_function(global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([global_data, global_data_feature], data_feature_content_threshold)

        while selection_round * self.selection_frequency <= max_steps and env.continue_run:
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
                state_data = env.state_data
                state_data_feature = self.get_state_data_feature_function(global_data, state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([state_data, state_data_feature], data_feature_content_threshold)

                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                else:
                    heuristic_trajectory_str = "\n".join([f"-----\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for round, items in enumerate(heuristic_traject[-5:])])
                prompt_dict["discuss_round"] = str(selection_round)
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)
                prompt_dict["selection_frequency"] = self.selection_frequency
                prompt_dict["num_candidate_heuristics"] = self.num_candidate_heuristics

                prompt_dict["demo_heuristic_str"] = ",".join([f"heuristic_name_{i + 1}"for i in range(self.num_candidate_heuristics)])
                
                self.llm_client.load("heuristic_selection", prompt_dict)
                response = self.llm_client.chat()
                self.llm_client.dump(f"step_{selection_round}")

                candidate_heuristics = extract(response, key="Selected heuristic", sep=",")
                for heuristic in candidate_heuristics:
                    assert heuristic in self.heuristic_pool
                
                # TTS selection
                selected_heuristic_name = tts_bon(env, candidate_heuristics, self.heuristic_pool, self.selection_frequency, self.rollout_budget, self.problem)
                pre_status = env.get_observation()
                if pre_status:
                    for _ in range(self.selection_frequency):
                        env.run_heuristic(self.heuristic_functions[selected_heuristic_name])
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Selection Index": selection_round + 1,
                        "Heuristic": selected_heuristic_name,
                    }
                    for key in pre_status.keys():
                        heuristic_dict["Delta of " + key] = cur_status[key] - pre_status[key]
                heuristic_traject.append(heuristic_dict)
                selection_round += 1
                # TODO: hidden rules
                # TODO: stop rules
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
