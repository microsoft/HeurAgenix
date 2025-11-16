import os
import pandas as pd

total_experiments = {"tsp":
    {
        "key_item": "current_cost",
        "upper_bound": [
            ("kroA100.tsp", 21282), ("kroA150.tsp", 26524), ("kroB100.tsp", 22141), ("kroB200.tsp", 29437), ("kroC100.tsp", 20749),
            ("bier127.tsp", 118282), ("tsp225.tsp", 3919), ("a280.tsp", 2579), ("pcb442.tsp", 50788), ("gr666.tsp", 294358),
            ("pa561.tsp", 2763), ("ts225.tsp", 126634), # ("pr1002.tsp", 259045), ("pr2392.tsp", 378032)
        ]
    }
}

experiments = [
    "gpt_4o.no_tc.no_reason", "gpt_4o.no_tc.reason", "gpt_4o.tc.no_reason", "gpt_4o.tc.reason",
    "gpt_5.no_tc.no_reason", "gpt_5.no_tc.reason", "gpt_5.tc.no_reason", "gpt_5.tc.reason",
    "meta-llama-3-8B-instruct.no_tc.no_reason", "meta-llama-3-8B-instruct.no_tc.reason", "meta-llama-3-8B-instruct.tc.no_reason", "meta-llama-3-8B-instruct.tc.reason",
    "qwen3-8B.no-think.no_tc.no_reason", "qwen3-8B.no-think.no_tc.reason", "qwen3-8B.no-think.tc.no_reason", "qwen3-8B.no-think.tc.reason",
    "qwen3-8B.think.no_tc.no_reason", "qwen3-8B.think.no_tc.reason", "qwen3-8B.think.tc.no_reason", "qwen3-8B.think.tc.reason",
    "qwen3-32B.no-think.no_tc.no_reason", "qwen3-32B.no-think.no_tc.reason", "qwen3-32B.no-think.tc.no_reason", "qwen3-32B.no-think.tc.reason",
    "qwen3-32B.think.no_tc.no_reason", "qwen3-32B.think.no_tc.reason", "qwen3-32B.think.tc.no_reason", "qwen3-32B.think.tc.reason"
]

def found_key(file_path: str, key_item: str) -> float:
    with open(file_path) as file:
        for line in file.readlines():
            if key_item in line.split(":")[0]:
                return float(line.split(":")[-1].strip())

def analysis_problem(problem: str, result_name: str, experiments: list[str]):
    key_item = total_experiments[problem]["key_item"]
    data_upper_bounds = total_experiments[problem]["upper_bound"]
    gaps = {data_upper_bound[0]: [] for data_upper_bound in data_upper_bounds}
    row_labels = [experiments]
    for experiment in experiments:
        for data, upper_bound in data_upper_bounds:
            output_dir = os.path.join("output", problem, result_name, data, experiment)
            if not os.path.exists(output_dir) or "finished.txt" not in os.listdir(output_dir):
                # print(f"Missing complete result in {output_dir}")
                gap = None
            else:
                result = found_key(os.path.join(output_dir, "result.txt"), key_item)
                if result is None:
                    # print(f"Missing complete result in {output_dir}")
                    gap = None
                else:
                    gap = (result - upper_bound) / upper_bound
                    gap = f"{round(gap * 100, 2)}%"
            gaps[data].append(gap)
    summary_file = os.path.join("output", problem, result_name, "summary.csv")
    print(f"Summary to {summary_file}")
    pd.DataFrame(gaps, index=row_labels).to_csv(summary_file)
    


analysis_problem("tsp", "result.20251106", experiments)