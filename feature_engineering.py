
import os
import importlib
import pandas as pd
from io import StringIO
from src.util.util import load_heuristic

problem = "tsp"
def feature_engineering(problem: str, sample_result_dirs: list[str], output_dir: str, global_data_feature_function: callable=None, state_data_feature_function:callable=None):
    module = importlib.import_module(f"src.problems.{problem}.components")
    module_dict = vars(module)
    globals().update(module_dict)
    module = importlib.import_module(f"src.problems.{problem}.env")
    module_dict = vars(module)
    globals().update(module_dict)
    if os.path.exists(os.path.join("output", problem, "evaluation_function.py")):
        evaluation_function_file = os.path.join("output", problem, "evaluation_function.py")
    elif os.path.exists(os.path.join("output", problem, "generate_evaluation_function", "evaluation_function.py")):
        evaluation_function_file = os.path.join("output", problem, "generate_evaluation_function", "evaluation_function.py")
    elif os.path.exists(os.path.join("src", "problems", problem, "evaluation_function.py")):
        evaluation_function_file = os.path.join("src", "problems", problem, "evaluation_function.py")
    if global_data_feature_function is None:
        global_data_feature_function = load_heuristic(evaluation_function_file, function_name="get_global_data_feature")
    if state_data_feature_function is None:
        state_data_feature_function = load_heuristic(evaluation_function_file, function_name="get_state_data_feature")

    os.makedirs(output_dir, exist_ok=True)
    source_file = open(os.path.join(output_dir, "source_data.txt"), "w")
    total_features = []
    heuristic_dir = os.path.join("src", "problems", "tsp", "heuristics", "basic_heuristics")
    for sample_result_dir in sample_result_dirs:
        information_file = open(os.path.join(sample_result_dir, "information.txt"))
        instance_name = information_file.readlines()[1].strip().split(": ")[-1].split("/")[-1]
        information_file.close()
        env = Env(instance_name)
        env.reset()

        max_construction_steps = env.construction_steps
        if not os.path.exists(os.path.join(sample_result_dir, f"round_{max_construction_steps * 2 - 1}.txt")):
            print(sample_result_dir)
            continue

        source_file.write(sample_result_dir + "\n")
        round_i = 0
        global_feature = global_data_feature_function(env.global_data)
        while os.path.exists(os.path.join(sample_result_dir, f"round_{round_i}.txt")):
            data_file = os.path.join(sample_result_dir, f"round_{round_i}.txt")
            data_content = open(data_file).readlines()
            if len(data_content) <= 2 or data_content[-2].split(":")[0] != "best_heuristics":
                break
            if set(eval(data_content[-1].split(":")[-1])) == {"False"}:
                break
            if round_i > 0:
                last_operator = ", ".join(data_content[round_i].strip().split(", ")[1:])
                env.run_operator(eval(last_operator))
            state_feature = state_data_feature_function(env.global_data, env.state_data)

            data = StringIO(open(data_file).read().split("---------------")[1])
            df = pd.read_csv(data, sep='\t')
            df = df.sort_values(by=["heuristic"])
            df["results"] = df["results"].apply(lambda x: sum([float(s) for s in x.split(",")]) / len(x.split(",")))
            min_ = min(df["results"])
            max_ = max(df["results"])
            df["results"] = df["results"].apply(lambda x: 1 - (x - min_) / (max_ - min_))
            heuristics = df["heuristic"].to_list()
            for name in heuristics:
                heuristic = load_heuristic(name, heuristic_dir)
                op = env.run_heuristic(heuristic, inplace=False)
                if op is False:
                    df.loc[df['heuristic'] == name, "results"] = 0

            ylabel = pd.Series(df.results.values, index=df.heuristic).to_dict()

            feature = global_feature | state_feature | dict(sorted(ylabel.items()))
            total_features.append(feature)
            round_i += 1

    df = pd.DataFrame(total_features)
    df.to_csv(os.path.join(output_dir, "feature.tsv"), index=False, sep="\t")
    source_file.close()

sample_result = os.path.join("output", "tsp", "heuristic_selection_data_collection", "small_instance_from_tsplib.20241112")
sample_result_dirs = [os.path.join(sample_result, sample_result_file) for sample_result_file in os.listdir(sample_result)]
# sample_result_dirs = [os.path.join(sample_result, "tsplib.bayg29.tsp.20241112_051151.result")]
feature_dir = sample_result + ".feature"
feature_engineering("tsp", sample_result_dirs, feature_dir)

sample_result = os.path.join("output", "tsp", "heuristic_selection_data_collection", "generated.20241112")
sample_result_dirs = [os.path.join(sample_result, sample_result_file) for sample_result_file in os.listdir(sample_result)]
feature_dir = sample_result + ".feature"
feature_engineering("tsp", sample_result_dirs, feature_dir)