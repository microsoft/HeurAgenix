import os
import traceback
import math
import numpy as np
from collections import Counter
from src.problems.base.env import BaseEnv
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
from src.util.util import extract, filter_dict_to_str, search_file

class MultiAgentLLMSelectionHyperHeuristic(LLMSelectionHyperHeuristic):
    def __init__(
        self,
        llm_clients: list[BaseLLMClient],
        heuristic_pool: list[str],
        problem: str,
        tool_calling: bool=False,
        iterations_scale_factor: float=2.0,
        selection_frequency: int=5,
        num_candidate_heuristics: int=3,
        rollout_budget: int=10,
        problem_state_content_threshold: int=1000,
    ) -> None:
        super().__init__(
            llm_client=llm_clients[0],
            heuristic_pool=heuristic_pool,
            problem=problem,
            tool_calling=tool_calling,
            iterations_scale_factor=iterations_scale_factor,
            selection_frequency=selection_frequency,
            num_candidate_heuristics=num_candidate_heuristics,
            rollout_budget=rollout_budget,
            problem_state_content_threshold=problem_state_content_threshold
        )
        self.llm_clients = llm_clients

    def update_prompt(self, env: BaseEnv, prompt_dict: dict, solution_problem_state: dict, selection_round: int, heuristic_traject: list) -> dict:
        solution_data = {"current_solution": env.current_solution, env.key_item: env.key_value}
        new_prompt_dict = prompt_dict.copy()
        new_prompt_dict["solution_problem_state"] = filter_dict_to_str([solution_data, solution_problem_state], self.problem_state_content_threshold)
        new_prompt_dict["discuss_round"] = str(selection_round)
        if heuristic_traject == []:
            heuristic_trajectory_str = "None"
        else:
            heuristic_trajectory_str = "\n".join([f"-----\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for items in heuristic_traject[-5:]])
        new_prompt_dict["heuristic_traject"] = heuristic_trajectory_str
        return new_prompt_dict

    def estimate_state_value(self, env: BaseEnv, prompt_dict: dict) -> float:
        client = self.llm_clients[0]
        # Load system prompt
        system_prompt_file = os.path.join("src", "problems", "base", "prompt", "system_prompt.txt")
        system_prompt = open(system_prompt_file, encoding="UTF-8").read()
        client.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        client.load("heuristic_selection", prompt_dict)
        response = client.chat()
        return 0

    def run(self, env: BaseEnv) -> bool:
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        max_rounds = math.ceil(max_steps / self.selection_frequency)
        selection_round = 0
        heuristic_traject = []
        instance_data = env.instance_data
        instance_problem_state = self.get_instance_problem_state(instance_data)

        # Init prompt dict
        prompt_dict = {}
        prompt_dict["problem"] = self.problem
        prompt_dict["problem_description"] = open(search_file("problem_description.txt", self.problem), encoding="utf-8").read()
        prompt_dict["instance_problem_state"] = filter_dict_to_str([instance_data, instance_problem_state], self.problem_state_content_threshold)
        prompt_dict["max_steps"] = max_steps
        prompt_dict["selection_frequency"] = self.selection_frequency
        prompt_dict["max_rounds"] = max_rounds
        prompt_dict["num_candidate_heuristics"] = self.num_candidate_heuristics
        prompt_dict["demo_heuristic_str"] = f"A/B/.../{self.last_heuristic_id}"
        prompt_dict["heuristic_pool_introduction"] = self.heuristic_pool_doc

        while selection_round <= max_rounds and env.continue_run:
            try:
                if env.is_complete_solution:
                    env.dump_result()

                # Generate state heuristic value
                solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)
                prompt_dict = self.update_prompt(env, prompt_dict, solution_problem_state=solution_problem_state, selection_round=selection_round, heuristic_traject=heuristic_traject)
                observation = self.get_observation_problem_state(solution_problem_state)
                observation[env.key_item] = env.key_value

                # Ask all agents for their initial choice
                candidate_heuristic_ids = []
                for i, client in enumerate(self.llm_clients):
                    # Load system prompt
                    system_prompt_file = os.path.join("src", "problems", "base", "prompt", "system_prompt.txt")
                    system_prompt = open(system_prompt_file, encoding="UTF-8").read()
                    client.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
                    client.load("heuristic_selection", prompt_dict)
                    try:
                        response_content = client.chat()
                        client.dump(f"step_{selection_round}_agent_{i}_initial")
                        
                        heuristic_id = extract(response_content, key="Selected heuristic id")
                        if heuristic_id and "[" in heuristic_id:
                            heuristic_id = heuristic_id.replace("[", "").replace("]", "").replace(" ", "").strip()
                            candidate_heuristic_ids.append(heuristic_id)
                    except Exception as e:
                        print(f"Agent {i} failed initial choice: {e}")

                # Check consensus
                selected_heuristic_id = None                
                if len(set(candidate_heuristic_ids)) == 1:
                    # Fast path: Perfect consensus
                    selected_heuristic_id = candidate_heuristic_ids[0]
                    print(f"Round {selection_round}: Perfect consensus. Selected {selected_heuristic_id}")
                else:
                    candidate_actions = list(set(candidate_heuristic_ids))
                    best_action_value = -1.0
                    best_action_id = None
                    for action_id in candidate_actions:
                        # Simulate action
                        sim_env = env.copy()
                        heuristic_func = self.heuristic_functions[action_id]
                        
                        # Run heuristic for selection_frequency steps
                        for _ in range(self.selection_frequency):
                            sim_env.run_heuristic(heuristic_func)

                        # Calculate simulated trajectory entry
                        sim_next_solution_problem_state = self.get_solution_problem_state(instance_data, sim_env.current_solution)
                        sim_next_observation = self.get_observation_problem_state(sim_next_solution_problem_state)
                        sim_next_observation[env.key_item] = sim_env.key_value
                        heuristic_dict = {
                            "Selection Index": selection_round,
                            "Selected heuristic ID": action_id,
                            "Heuristic": self.heuristic_names[action_id]
                        }
                        for key in observation.keys():
                            heuristic_dict["Delta of " + key] = f"From {observation[key]} to {sim_next_observation[key]}"
                        sim_prompt_dict = self.update_prompt(sim_env, prompt_dict, sim_next_solution_problem_state, selection_round=selection_round, heuristic_traject=heuristic_traject + [heuristic_dict])
                        

                        # Estimate value of new state
                        state_value = self.estimate_state_value(sim_env, sim_prompt_dict)

                        if state_value > best_action_value:
                            best_action_value = state_value
                            best_action_id = action_id
                            
                    selected_heuristic_id = best_action_id
                    print(f"Round {selection_round}: Lookahead selected {selected_heuristic_id} with V(s')={best_action_value:.2f}")

                selected_heuristic_name = self.heuristic_names[selected_heuristic_id]
                selected_heuristic = self.heuristic_functions[selected_heuristic_id]
                # Record selection and observation
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
                for key in observation.keys():
                    heuristic_dict["Delta of " + key] = f"From {observation[key]} to {next_observation[key]}"
                heuristic_traject.append(heuristic_dict)
                selection_round += 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
