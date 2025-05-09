import os
import traceback
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.compare_heuristics import best_result


class LLMSelectionHyperHeuristicTTS:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        heuristic_pool: list[str],
        problem: str,
        search_time: int=10,
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.heuristic_docs = {
            heuristic_file.split(".")[0]: extract_function_with_short_docstring(open(search_file(heuristic_file, problem)).read(), heuristic_file.split(".")[0]) 
            for heuristic_file in heuristic_pool}
        self.heuristic_pools = {
            heuristic_file.split(".")[0]: load_heuristic(heuristic_file, problem=self.problem)
            for heuristic_file in heuristic_pool}
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")
        self.search_time = search_time

    def run(self, env:BaseEnv,  max_steps: int=None, data_feature_content_threshold: int=1000, **kwargs) -> bool:
        # Load background
        prompt_dict = self.llm_client.load_background(self.problem)

        # Load heuristic pool
        max_steps = max_steps if max_steps is not None else env.construction_steps * 3
        prompt_dict["heuristic_pool_introduction"] = "\n".join(self.heuristic_docs.values())
        self.llm_client.load("heuristic_pool", prompt_dict)
        self.llm_client.chat()
        self.llm_client.dump("heuristic_pool")

        # Generate global heuristic value
        global_data = env.global_data
        global_data_feature = self.get_global_data_feature_function(global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([global_data, global_data_feature], data_feature_content_threshold)

        heuristic_traject = []
        current_steps = 0
        running_step = 5
        while current_steps <= max_steps and env.continue_run:
            try:
                if env.is_complete_solution:
                    env.dump_result()
                self.llm_client.load_chat("heuristic_pool")

                # Generate state heuristic value
                state_data = env.state_data
                state_data_feature = self.get_state_data_feature_function(global_data, state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([state_data, state_data_feature], data_feature_content_threshold)

                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                    last_heuristic = "None"
                else:
                    heuristic_trajectory_str = "\n".join([f"---Round {round}---\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for round, items in enumerate(heuristic_traject)])
                    last_heuristic = heuristic_traject[-1]["Heuristic"]
                    
                prompt_dict["discuss_round"] = str(len(heuristic_traject))
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["last_heuristic"] = last_heuristic
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)
                self.llm_client.load("heuristic_selection_tts", prompt_dict)

                response = self.llm_client.chat()
                self.llm_client.dump(f"step_{len(heuristic_traject)}")

                if "Run heuristic:" in response:
                    # Load selected heuristic, running step, parameters(optional) and reason
                    result = extract(response, key="Run heuristic", sep="\n")
                    selected_heuristic_names = result[0].split(":")[-1].strip()
                    # running_step = int(result[1].split(":")[-1].strip().split(" ")[0])
                    explain = result[-1].split(":")[-1].strip()
                    parameters = {}

                    candidate_heuristics = [h for h in selected_heuristic_names.replace(" ", "").split(";") if h in self.heuristic_pools]

                    selected_heuristic_name = best_result(env, candidate_heuristics, self.heuristic_pools.keys(), self.search_time, self.problem, best_result_proxy)

                    assert selected_heuristic_name in self.heuristic_pools.keys()
                    selected_heuristic = self.heuristic_pools[selected_heuristic_name]

                    pre_status = env.get_observation()
                    if pre_status:
                        for _ in range(running_step):
                            env.run_heuristic(selected_heuristic, parameters=parameters)
                        cur_status = env.get_observation()
                        heuristic_dict = {
                            "Heuristic": selected_heuristic_name,
                            # "Parameters": parameters,
                            "Running Steps": running_step,
                            "Explain": explain
                        }
                        for key in pre_status.keys():
                            heuristic_dict["Delta of " + key] = cur_status[key] - pre_status[key]
                    if len(heuristic_traject) >= 5:
                        heuristic_traject.pop(0)
                    heuristic_traject.append(heuristic_dict)
                    current_steps += running_step
                elif "Stop" in response or "None" in response:
                    if env.is_complete_solution:
                        break
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
