import os
import math
import numpy as np
import matplotlib.pyplot as plt

total_experiments = [
    {
        "problem": "tsp",
        "key_item": "current_cost",
        "data": ["tsp225.tsp", "a280.tsp", "pcb442.tsp", "pa561.tsp", "gr666.tsp", "pr1002.tsp", "pr2392.tsp"],
        "heuristics": ["nearest_neighbor_f91d", "nearest_neighbor_e8a4", "cheapest_insertion_605f", "cheapest_insertion_7a30", "farthest_insertion_b6d3", "farthest_insertion_54db"],
        "upper_bound": [3919, 2579, 50788, 2763, 294358, 259045, 378032]
    },
    {
        "problem": "cvrp",
        "key_item": "total_current_cost",
        "data": ["A-n80-k10.vrp", "B-n78-k10.vrp", "E-n101-k14.vrp", "F-n135-k7.vrp", "M-n200-k17.vrp", "P-n101-k4.vrp", "X-n1001-k43.vrp"],
        "heuristics": ["nearest_neighbor_99ba", "nearest_neighbor_54a9", "min_cost_insertion_7bfa", "min_cost_insertion_3b2b", "farthest_insertion_ce2b", "farthest_insertion_6308",],
        "upper_bound": [1762, 1221, 1067, 1162, 1275, 681, 72355]
    },
    {
        "problem": "jssp",
        "key_item": "current_makespan",
        "data": ["LA05.jssp", "LA10.jssp", "LA15.jssp", "LA20.jssp", "LA25.jssp", "LA30.jssp", "LA35.jssp"],
        "heuristics": ["most_work_remaining_930e", "most_work_remaining_df20", "first_come_first_served_6c4f", "first_come_first_served_af26", "shortest_processing_time_first_c374", "shortest_processing_time_first_d471"],
        "upper_bound": [593, 958, 1207, 902, 977, 1355, 1888]
    },
    {
        "problem": "max_cut",
        "key_item": "current_cut_value",
        "data": ["g10.mc", "g20.mc", "g30.mc", "toursg3-8.mc", "toursg3-15.mc", "tourspm3-8-50.mc", "tourspm3-15-50.mc"],
        "heuristics": ["most_weight_neighbors_320c", "most_weight_neighbors_d31b", "highest_weight_edge_eb0c", "highest_weight_edge_ca02", "balanced_cut_21d5", "balanced_cut_c0e6"],
        "upper_bound": [1994, 941, 3403, 41684814, 281029888, 454, 2964]
    },
    {
        "problem": "mkp",
        "key_item": "current_profit",
        "data": ["mknap1_1.mkp", "mknap1_7.mkp", "WEING1.DAT.mkp", "PB7.DAT.mkp", "mknapcb9-01.mkp", "mknapcb9-11.mkp", "mknapcb9-21.mkp"],
        "heuristics": ["greedy_by_profit_8df3", "greedy_by_profit_1597", "greedy_by_weight_ece2", "greedy_by_weight_e7f9", "greedy_by_density_9e8d", "greedy_by_density_bb0a"],
        "upper_bound": [3800, 16537, 141278, 1035, 115868, 217995, 301627]
    },
    {
        "problem": "dposp",
        "key_item": "fulfilled_order_num",
        "data": ["test_case_0", "test_case_1", "test_case_2", "test_case_3", "test_case_4", "test_case_5", "test_case_6"],
        "heuristics": ["least_order_remaining_9c3c", "least_order_remaining_27ca", "shortest_operation_ff40", "shortest_operation_ae31", "greedy_by_order_density_c702", "greedy_by_order_density_de77"],
        "upper_bound": [None, None, None, None, None, None, None]
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
        data = problem_dict["data"]
        heuristics = problem_dict["heuristics"]
        upper_bounds = problem_dict["upper_bound"]
        for data_index, data in enumerate(data):
            upper_bound = upper_bounds[data_index]
            for heuristic in heuristics:
                result_file = os.path.join("output", problem, "result", data, heuristic, "result.txt")
                if os.path.exists(result_file):
                    value = found_key(result_file, key_item)
                    if value:
                        gap = "None" if upper_bound is None else f"{round(abs(value - upper_bound) / upper_bound * 100, 2)}%"
                        print(problem, data, heuristic, key_item, value, gap)
                else:
                    print(F"Missing {result_file}")
                    
dump_all_result()