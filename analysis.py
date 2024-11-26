import os
import math
import numpy as np
import matplotlib.pyplot as plt

total_experiments = [
    {
        "problem": "tsp",
        "key_item": "current_cost",
        "data": [("tsp225.tsp", 3919), ("a280.tsp", 2579), ("pcb442.tsp", 50788), ("pa561.tsp", 2763), ("gr666.tsp", 294358), ("pr1002.tsp", 259045), ("pr2392.tsp", 378032)],
        "experiments": ["nearest_neighbor_f91d", "cheapest_insertion_605f", "farthest_insertion_b6d3"],
    },
    {
        "problem": "cvrp",
        "key_item": "total_current_cost",
        "data": [("A-n80-k10.vrp", 1762), ("B-n78-k10.vrp", 1221), ("E-n101-k14.vrp", 1067), ("F-n135-k7.vrp", 1162), ("M-n200-k17.vrp", 1275), ("P-n101-k4.vrp", 681), ("X-n1001-k43.vrp", 72355)],
        "experiments": ["nearest_neighbor_99ba", "min_cost_insertion_7bfa", "farthest_insertion_ce2b"],
    },
    {
        "problem": "jssp",
        "key_item": "current_makespan",
        "data": [("LA05.jssp", 593), ("LA10.jssp", 958), ("LA15.jssp", 1207), ("LA20.jssp", 902), ("LA25.jssp", 977), ("LA30.jssp", 1355), ("LA35.jssp", 1888)],
        "experiments": ["most_work_remaining_930e", "first_come_first_served_6c4f", "shortest_processing_time_first_c374"],
    },
    {
        "problem": "max_cut",
        "key_item": "current_cut_value",
        "data": [("g10.rud", 1994), ("g20.rud", 941), ("g30.rud", 3403), ("toursg3-8.txt", 41684814), ("toursg3-15.txt", 281029888), ("tourspm3-8-50.txt", 454), ("tourspm3-15-50.txt", 2964)],
        "experiments": ["most_weight_neighbors_320c", "highest_weight_edge_eb0c", "balanced_cut_21d5"],
    },
    { 
        "problem": "mkp",
        "key_item": "current_profit",
        "data": [("mknap1_1.mkp", 3800), ("mknap1_7.mkp", 16537), ("WEING1.DAT.mkp", 141278), ("PB7.DAT.mkp", 1035), ("mknapcb9-01.mkp", 115868), ("mknapcb9-11.mkp", 217995), ("mknapcb9-21.mkp", 301627)],
        "experiments": ["greedy_by_profit_8df3", "greedy_by_weight_ece2", "greedy_by_density_9e8d"]
    },
    {
        "problem": "dposp",
        "key_item": "fulfilled_order_num",
        "data": [("test_case_1", 0), ("test_case_2", 0), ("test_case_3", 0), ("test_case_4", 0), ("test_case_5", 0), ("test_case_6", 0), ("test_case_7", 0)],
        "experiments": ["nearest_order_scheduling_1a5e", "greedy_deadline_proximity_ac6e", "greedy_by_order_density_c702"]
    }
]

def found_key(file_path: str, key_item: str) -> float:
    with open(file_path) as file:
        for line in file.readlines():
            if key_item in line.split(":")[0]:
                return float(line.split(":")[-1].strip())

def dump_single_to_latex():
    latex_strs = ["\\toprule\nProblem & Heuristic & \\multicolumn{7}{c}{Data}"]
    for problem_dict in total_experiments:
        problem = problem_dict["problem"]
        key_item = problem_dict["key_item"]
        latex_strs += ["\midrule\n\multirow{8}{*}{" + problem + "} & & " + " & ".join([data_name for data_name, upper_bound in problem_dict["data"]])]
        for experiment in problem_dict["experiments"]:
            gap_strs = []
            for data_name, upper_bound in problem_dict["data"]:
                result_file = os.path.join("output", problem, "result", data_name, experiment, "result.txt")
                value = found_key(result_file, key_item)
                gap = int(value) if upper_bound == 0 else round((abs(value - upper_bound) / upper_bound) * 100, 2)
                gap_strs.append(str(gap))
            latex_strs += [f"& {experiment} & " + " & ".join(gap_strs)]
    print(" \\\\ \n".join(latex_strs) + " \\\\")

def dump_hh_to_latex():
    latex_strs = ["\\toprule\nProblem & Heuristic & \\multicolumn{7}{c}{Data}"]
    for problem_dict in total_experiments:
        problem = problem_dict["problem"]
        key_item = problem_dict["key_item"]
        latex_strs += ["\midrule\n\multirow{8}{*}{" + problem + "} & & " + " & ".join([data_name for data_name, upper_bound in problem_dict["data"]])]
        for experiment in ["gpt_hh", "random_hh"]:
            gap_strs = []
            for data_name, upper_bound in problem_dict["data"]:
                gaps = []
                for single_experiment in os.listdir(os.path.join("output", problem, "result", data_name)):
                    if single_experiment.split(".")[0] == experiment:
                        result_file = os.path.join("output", problem, "result", data_name, single_experiment, "result.txt")
                        if os.path.exists(result_file):
                            value = found_key(result_file, key_item)
                            gap = int(value) if upper_bound == 0 else (abs(value - upper_bound) / upper_bound) * 100
                            gaps.append(gap)
                        else:
                            print(F"Missing {result_file}")
                if gaps == []:
                    gap_strs.append("None")
                else:
                    mean = round(sum(gaps) / len(gaps), 2)
                    std = round(math.sqrt(sum((gap - mean) ** 2 for gap in gaps) / len(gaps)), 2)
                    gap_strs.append(f"{mean}$\pm${std}")
            latex_strs += [f"& {experiment} & " + " & ".join(gap_strs)]
    print(" \\\\ \n".join(latex_strs) + " \\\\")

def dump_single_to_image():
    bar_width = 0.15
    colors = ['lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkred']
    for problem_dict in total_experiments:
        problem = problem_dict["problem"]
        key_item = problem_dict["key_item"]
        datas = [data[0].split(".")[0] for data in problem_dict["data"]]
        experiments = problem_dict["experiments"]
        data_num = len(problem_dict["data"])
        experiment_num = len(experiments)
        results = np.zeros((data_num, experiment_num))
        for exp_index, experiment in enumerate(experiments):
            for data_index, (data_name, upper_bound) in enumerate(problem_dict["data"]):
                result_file = os.path.join("output", problem, "result", data_name, experiment, "result.txt")
                value = found_key(result_file, key_item)
                gap = int(value) if upper_bound == 0 else round((abs(value - upper_bound) / upper_bound) * 100, 2)
                results[data_index, exp_index] = gap

        indices = np.arange(data_num)
        plt.figure(figsize=(25, 4))
        for exp_index, experiment in enumerate(experiments):
            experiment = experiment[:-5]
            experiment = experiment.replace("_", " ")
            bars = plt.bar(indices + exp_index * bar_width, results[:, exp_index], width=bar_width, color=colors[exp_index], label=experiment)
            for data_index, bar in enumerate(bars):
                yval = bar.get_height() + 0.2
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(results[data_index, exp_index], 2), ha='center', va='bottom', fontsize=7)

        plt.xlim([min(indices) - bar_width, max(indices) + experiment_num * bar_width])
        plt.xticks(indices + bar_width * (experiment_num - 1) / 2, datas)
        plt.legend()
        plt.title(problem)
        plt.ylabel('Gap to best result')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False) 
        legend = plt.legend()
        legend.set_draggable(True)
        plt.savefig("evo_" + problem)

def dump_hh_to_image():
    bar_width = 0.15
    colors = ['lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkred']
    experiments = ["gpt_hh", "gpt_hh_evolved", "random_hh", "random_hh_evolved"]
    experiment_names = ["LLM selection (basic)", "LLM selection (evolved)", "Random selection (basic)", "Random selection (evolved)"]
    for problem_dict in total_experiments:
        problem = problem_dict["problem"]
        key_item = problem_dict["key_item"]
        datas = problem_dict["data"]
        data_num = len(datas)
        experiment_num = len(experiments)
        means = np.zeros((data_num, experiment_num))
        sems = np.zeros((data_num, experiment_num))
        for exp_index, experiment in experiments:
            for data_index, (data_name, upper_bound) in enumerate(problem_dict["data"]):
                gaps = []
                for single_experiment in os.listdir(os.path.join("output", problem, "result", data_name)):
                    if single_experiment.split(".")[0] == experiment:
                        result_file = os.path.join("output", problem, "result", data_name, single_experiment, "result.txt")
                        if os.path.exists(result_file):
                            value = found_key(result_file, key_item)
                            gap = int(value) if upper_bound == 0 else (abs(value - upper_bound) / upper_bound) * 100
                            gaps.append(gap)
                        else:
                            print(F"Missing {result_file}")
                if gaps == []:
                    raise "Missing"
                else:
                    mean = round(sum(gaps) / len(gaps), 2)
                    std = math.sqrt(sum((gap - mean) ** 2 for gap in gaps) / len(gaps))
                    sem = round(std / math.sqrt(len(gaps)), 2)
                    means[data_index, exp_index] = mean
                    sems[data_index, exp_index] = sem

        plt.figure(figsize=(25, 4))
        indices = np.arange(data_num) 

        for exp_index, experiment_name in enumerate(experiment_names):
            bars = plt.bar(indices + exp_index * bar_width, means[:, exp_index], width=bar_width, color=colors[exp_index], label=experiment_name, yerr=sems[:, exp_index], capsize=5)  
            for bar, mean, var in zip(bars, means[:, exp_index], sems[:, exp_index]):  
                yval = bar.get_height() + var + 0.2
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{mean:.2f}\n±{var:.2f}', ha='center', va='bottom', fontsize=7)  

        plt.xlim([min(indices) - bar_width, max(indices) + bar_width * experiment_num])  
        plt.xticks(indices + bar_width * (experiment_num - 1) / 2, datas)  
        ax = plt.gca()
        ax.legend(loc='upper right', bbox_to_anchor=(1.13, 0.8), bbox_transform=ax.transAxes)  

        plt.title(problem)
        plt.ylabel('Gap to best result')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False) 
        plt.savefig("selection_" + problem)


def dump_all_latex():
    dump_single_to_latex()
    dump_hh_to_latex()

def dump_all_image():
    dump_single_to_image()
    dump_hh_to_image()

def dump_all_result():
    for problem_dict in total_experiments:
        problem = problem_dict["problem"]
        key_item = problem_dict["key_item"]
        for data in os.listdir(os.path.join("output", problem, "result")):
            for experiment in os.listdir(os.path.join("output", problem, "result", data)):
                result_file = os.path.join("output", problem, "result", data, experiment, "result.txt")
                if os.path.exists(result_file):
                    value = found_key(result_file, key_item)
                    if value:
                        print(problem, data, experiment, key_item, value)
                else:
                    print(F"Missing {result_file}")
                    
dump_all_result()