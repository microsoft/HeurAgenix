import os
import traceback
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str
from src.util.gpt_helper import GPTHelper
import re
from collections import Counter
from find_best import find_best

refinement_names = [
    "simulated_annealing_e625",
    "_2opt_89aa",
    "_3opt_e75b",
    "three_opt_e8d7",
    "two_opt_0554",
    "node_shift_between_routes_7b8a",
]

class GPTSelectionHyperHeuristic:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        heuristic_dir: str,
        problem: str,
        use_CPR_mode: bool = False,
    ) -> None:
        self.gpt_helper = gpt_helper
        self.use_CPR_mode = use_CPR_mode
        self.problem = problem
        self.heuristic_docs = {
            heuristic_file.split(".")[0]: extract_function_with_short_docstring(open(os.path.join(heuristic_dir, heuristic_file)).read(), heuristic_file.split(".")[0]) 
            for heuristic_file in os.listdir(heuristic_dir)}
        self.heuristic_pools = {
            heuristic_file.split(".")[0]: load_heuristic(heuristic_file, problem=self.problem)
            for heuristic_file in os.listdir(heuristic_dir)}
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")

    def run(self, env:BaseEnv, max_steps: int=None, data_feature_content_threshold: int=1000, **kwargs) -> bool:
        # # Load background
        prompt_dict = self.gpt_helper.load_prompt_dict(self.problem)

        max_steps = max_steps if max_steps is not None else env.construction_steps * 2

        # Generate global heuristic value
        global_data = env.global_data
        global_data_feature = self.get_global_data_feature_function(global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([global_data, global_data_feature], data_feature_content_threshold)

        disabled_algorithms_str = ''

        heuristic_traject = []
        current_steps = 0
        tsp_name = env.data_name.split('/')[-1]
        tsp_name = tsp_name.split(".")[0]

        while current_steps <= max_steps or not env.is_complete_solution:
            try:
                self.gpt_helper.reset()
                # Generate state heuristic value
                state_data = env.state_data
                state_data_feature = self.get_state_data_feature_function(global_data, state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([state_data, state_data_feature], data_feature_content_threshold)
                total = 0
                disabled_algorithms = set()
                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                    last_heuristic = "None"
                else:
                    total = len(heuristic_traject)
                    if total > 5:
                        traj_to_show = enumerate(heuristic_traject[-5:], start=total - 5)
                    else:
                        traj_to_show = enumerate(heuristic_traject)
                        
                    heuristic_trajectory_str = "\n".join(
                        [f"---Round {round}---\n" + "\n".join(f"{key}: {value}" for key, value in items.items())
                        for round, items in traj_to_show]
                    )
                    last_heuristic = heuristic_traject[-1]["Heuristic"]

                prompt_dict["discuss_round"] = total
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["last_heuristic"] = last_heuristic
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)


                usr_prompt = f"""Global Data (Partial heuristic metrics):
- Node Count: {prompt_dict['node_num']}
- Average Distance: {global_data_feature['average_distance']}
- Minimum Distance: {global_data_feature['min_distance']}
- Maximum Distance: {global_data_feature['max_distance']}
- Distance Standard Deviation: {global_data_feature['std_dev_distance']}
- Node Density: {global_data_feature['density']}
- Centroid: {global_data_feature['centroid']}
Note: Some data are omitted due to space constraints.

Current Stage State Data:
- Visited Nodes: {state_data_feature['current_path_length']}
- Unvisited Nodes: {state_data_feature['unvisited_num']}
- Current Total Cost: {state_data_feature['current_cost']}
- Last Edge Cost: {state_data_feature['last_edge_cost']}
- Current Path Length: {state_data_feature['current_path_length']}
- Remaining Nodes: {state_data_feature['remaining_nodes']}
- Average Edge Cost: {state_data_feature['average_edge_cost']}
- Edge Cost Standard Deviation: {state_data_feature['std_dev_edge_cost']}
- Solution Validity: {state_data_feature['solution_validity']}
- Minimum Remaining Edge Cost: {state_data_feature['min_edge_cost_remaining']}
- Maximum Remaining Edge Cost: {state_data_feature['max_edge_cost_remaining']}
Note: Some data are omitted due to space constraints.

Discussion History:
We have already completed {total} rounds of discussion. Summary:
{heuristic_trajectory_str}

Explanation:
- "Delta of Visited Node" indicates the change in the number of newly visited nodes this round.
- "Delta of Current Cost" indicates the additional cost incurred by the applied heuristic; a smaller (or more negative) value suggests a better improvement.

"""
                if self.use_CPR_mode:
                    with open("src/problems/tsp/prompt/heuristic_pool_CPR.txt", "r", encoding="utf-8") as file:
                        system_prompt = file.read()
                else:
                    with open("src/problems/tsp/prompt/heuristic_pool.txt", "r", encoding="utf-8") as file:
                        system_prompt = file.read()

                response = self.gpt_helper.chat(system_prompt,usr_prompt)
                response = response.replace("\\n", "\n")
                self.gpt_helper.dump(f"step_{total}")
                
                
                if "Run heuristic:" in response:

                    pattern = r"selected heuristic\s*:\s*([^\s\n]+)"
                    matches = re.findall(pattern, response)

                    response_ans = []
                    for i in matches:
                        if i in self.heuristic_pools:
                            response_ans.append(i)
                        else:
                            partial_name = i.strip().split("_")[-1]
                            candidates = [name for name in self.heuristic_pools
                                        if partial_name in name]
                            response_ans.append(candidates[0])
                            if not candidates:
                                continue

                    print(f'response_ans:{response_ans}')
                    best_ans = find_best(env,response_ans,self.heuristic_pools,10,'tsp')
                    
                    if best_ans not in self.heuristic_pools:
                        continue

                    if best_ans in self.heuristic_pools:
                        actual_name = best_ans
                    else:
                        partial_name = best_ans.strip().split("_")[-1]
                        candidates = [name for name in self.heuristic_pools
                                    if partial_name in name]
                        if not candidates:
                            continue
                        actual_name = candidates[0]
                    
                    selected_heuristic = self.heuristic_pools[actual_name]

                    pre_status = env.get_observation()
                    tem_step = 3
                    for _ in range(tem_step):
                        print(f'Successfully run the heuristic:{actual_name}')
                        env.run_heuristic(selected_heuristic)
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Heuristic": actual_name,
                        # "Parameters": parameters,
                    }
                    for key in pre_status.keys():
                        heuristic_dict["Delta of " + key] = cur_status[key] - pre_status[key]
                    heuristic_traject.append(heuristic_dict)
                    current_steps += tem_step
                elif "Stop" in response or "None" in response:
                    if env.is_complete_solution:
                        break
                    else:
                        current_steps -= 1
                else:
                    current_steps -= 1
                    continue

            
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
