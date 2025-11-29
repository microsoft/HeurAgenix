import os
import random
import datetime
from src.util.util import extract
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import load_function
from src.problems.max_cut.env import Env
from collections import defaultdict, deque
from best_known import get_best
import numpy as np

def refine_code():
    llm_client = get_llm_client(os.path.join("data", "llm_config", "azure_gpt_5.json"), output_dir=os.path.join("output", "refine_code"))
    for heuristic_file in os.listdir(os.path.join("src", "problems", "max_cut", "heuristics", "basic_heuristics")):
        llm_client.load_chat("background.json")
        file = open(os.path.join("src", "problems", "max_cut", "heuristics", "basic_heuristics", heuristic_file), encoding="utf-8").read()
        llm_client.load("refine_code.txt", {"code": file})
        response = llm_client.chat()
        code = extract(response, "python_code")
        heuristic_name = code.split("def ")[1].split("(")[0]
        llm_client.dump(heuristic_name)
        with open(os.path.join("output", "refine_code", heuristic_name + ".py"), "w", encoding="utf-8") as f:
            f.write(code)
 
def dedup():
    llm_client = get_llm_client(os.path.join("data", "llm_config", "azure_gpt_5.json"), output_dir=os.path.join("output", "refine_code"))
    heuristic_files = os.listdir(os.path.join("src", "problems", "max_cut", "heuristics", "basic_heuristics"))
    num = len(heuristic_files)
    all_code = ""
    for heuristic_file in heuristic_files:
        code = open(os.path.join("src", "problems", "max_cut", "heuristics", "basic_heuristics", heuristic_file), encoding="utf-8").read()
        name = code.split("def ")[1].split("(")[0]
        all_code = all_code + f"------------------------------\n\n\n----------{name}----------\n\n" + code + "\n\n"
    llm_client.load("deduce_code.txt", {"num": num, "all_code": all_code})
    llm_client.chat()
    llm_client.dump("dedup3")
 
 
def generate_data(source_file: str, new_file: str):
    # 读取图
    new_node_num = random.randint(10, 100)
    with open(source_file, encoding="utf-8") as f:
        header_tokens = f.readline().strip().split()
        if not header_tokens:
            raise ValueError("Empty file or invalid header")
        if len(header_tokens) == 1:
            total_nodes_in = int(header_tokens[0])
            has_edge_count = False
        else:
            total_nodes_in = int(header_tokens[0])
            total_edges_in = int(header_tokens[1])
            has_edge_count = True

        # 邻接表与边权
        adj = defaultdict(set)
        edge_w = dict()  # key = (min(u,v), max(u,v)) -> weight

        min_id, max_id = None, None
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, v_str, w_str = line.split()
            u = int(u_str); v = int(v_str); w = int(w_str)

            if min_id is None:
                min_id, max_id = u, u
            min_id = min(min_id, u, v)
            max_id = max(max_id, u, v)

            if u == v:
                # 保留自环也不影响 BFS 选择，若不希望保留可 continue 掉
                pass
            adj[u].add(v)
            adj[v].add(u)
            a, b = (u, v) if u < v else (v, u)
            edge_w[(a, b)] = w  # 若有重复边，以最后一次为准

    # 检测输入节点基数：若出现 0 则默认 0 基，否则默认 1 基
    input_zero_based = (min_id == 0)

    # 仅考虑有边的节点（度>0），避免抽到孤点
    nodes_with_edges = set(adj.keys())
    if not nodes_with_edges:
        raise ValueError("Graph has no edges; cannot build connected subgraph")

    # 找最大连通分量
    visited = set()
    def bfs_component(start):
        comp = set()
        dq = deque([start])
        visited.add(start)
        while dq:
            x = dq.popleft()
            comp.add(x)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    dq.append(y)
        return comp

    components = []
    for node in nodes_with_edges:
        if node not in visited:
            comp_nodes = bfs_component(node)
            components.append(comp_nodes)

    # 最大连通分量
    lcc = max(components, key=len)
    if new_node_num > len(lcc):
        raise ValueError(
            f"Requested {new_node_num} nodes, but largest connected component "
            f"has only {len(lcc)} nodes. Reduce new_node_num or allow disconnected sampling."
        )

    # 选择起点：最大连通分量里度数最高的节点
    seed = max(lcc, key=lambda x: len(adj[x]))

    # 在最大连通分量内做 BFS/雪球扩展，优先访问高度邻居，提高稠密性
    selected = []
    selected_set = set()
    dq = deque([seed])
    selected.append(seed)
    selected_set.add(seed)

    while dq and len(selected) < new_node_num:
        u = dq.popleft()
        # 邻居按度数降序访问，倾向于更稠密区域
        neighbors = sorted((v for v in adj[u] if v in lcc and v not in selected_set),
                           key=lambda x: len(adj[x]), reverse=True)
        for v in neighbors:
            if v not in selected_set:
                selected.append(v)
                selected_set.add(v)
                dq.append(v)
                if len(selected) == new_node_num:
                    break

    # 诱导子图的边
    sub_edges = []
    for (a, b), w in edge_w.items():
        if a in selected_set and b in selected_set:
            sub_edges.append((a, b, w))

    # 重新映射节点为连续编号（按访问顺序），并保持输入的基数风格
    out_base = 0 if input_zero_based else 1
    mapping = {node: i + out_base for i, node in enumerate(selected)}

    # 写文件：头部风格尽量与输入保持一致
    with open(new_file, "w", encoding="utf-8") as f:
        if has_edge_count:
            f.write(f"{new_node_num} {len(sub_edges)}\n")
        else:
            f.write(f"{new_node_num}\n")
        for a, b, w in sub_edges:
            f.write(f"{mapping[a]} {mapping[b]} {w}\n")
   
 


def batch_evolved():
    for heuristic_file in os.listdir(os.path.join("src", "problems", "max_cut", "heuristics", "refined_basic_heuristics")):
        s = f"start \"\" /B python evolve_heuristic.py -p max_cut -m -l data\\llm_config\\azure_gpt_5.json -ed output\\max_cut\\generated_data -e {heuristic_file}"
        print(s)


 
def test_single_heuristic(target_heuristic: callable, heuristic_pools: list[callable], test_data_list: list[str], test_ratio: float=0.3):
    total_ms = 0
    nones = 0
    crashed = []
    running_steps = 0
    complete = 0
    total_gap = 0
    for test_data in test_data_list:
        env = Env(data_name=test_data)
        env.reset()
        f = open("time_cost.txt", "a", encoding="utf-8")
        f.write(f"---{target_heuristic.__name__}, {test_data}, {env.instance_data['node_num']}---\n")
        f.close()
        total_times = int(env.construction_steps * 2)
        test_times = int(env.construction_steps * 2 * test_ratio)
        test_steps = random.sample(range(total_times), test_times)
        test_step_flag = [1 if i in test_steps else 0 for i in range(total_times)]
        time_cost = {heuristic.__name__: [] for heuristic in heuristic_pools}
        for j in range(env.construction_steps * 2):
            if j % 500 == 0 or j == env.construction_steps * 2 - 1:
                f = open("time_cost.txt", "a", encoding="utf-8")
                time_cost_list = [(heuristic, np.mean(ms)) if len(ms) > 0 else (heuristic, 0) for heuristic, ms in time_cost.items()]
                time_cost_list = sorted(time_cost_list, key=lambda x: x[1], reverse=True)
                time_cost_str = ", ".join([f"{heuristic}: {ms:.2f}ms" for heuristic, ms in time_cost_list if ms > 500])
                time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{time}\t{j}\t{time_cost_str}\n")
                f.close()
                time_cost = {heuristic.__name__:[] for heuristic in heuristic_pools}
                
            if test_step_flag[j] == 1:
                begin = datetime.datetime.now()
                op = env.run_heuristic(target_heuristic)
                end = datetime.datetime.now()
                running_steps += 1
                if op is None:
                    nones += 1
                if isinstance(op, str):
                    crashed.append(op)
                total_ms += (end - begin).microseconds / 1000
            else:
                heuristic = random.choice(heuristic_pools)
                begin = datetime.datetime.now()
                env.run_heuristic(heuristic)
                end = datetime.datetime.now()
                ms = (end - begin).microseconds / 1000
                time_cost[heuristic.__name__].append(ms)
        if env.is_complete_solution and env.is_valid_solution:
            complete += 1
            best_known = get_best(test_data)
            total_gap += abs(env.key_value - best_known) / best_known
    return running_steps, total_ms / running_steps, nones / running_steps, complete / len(test_data_list), total_gap / len(test_data_list), "\n".join(crashed)

def test_all_heuristics(test_dir, test_data_list):
    test_data_names = ",".join(test_data_list)
    result_name = "comparison_test.txt"
    with open(result_name, "w", encoding="utf-8") as f:
        f.write(test_data_names + "\n")
        f.write("Heuristic\ttotal_running_steps\taverage_time_cost(ms)\taverage_nones\tcomplete_ratio\taverage_gap\tcrashed\n")
        f.close()
    heuristics_pool = []
    for heuristic_file in os.listdir(test_dir):
        heuristic = load_function(os.path.join(test_dir, heuristic_file), "max_cut")
        heuristics_pool.append(heuristic)
    for heuristic_file in heuristics_pool[5:]:
        copied_heuristics = heuristics_pool.copy()
        copied_heuristics.remove(heuristic_file)
        running_steps, average_ms, average_nones, complete_ratio, average_gap, crashed = \
            test_single_heuristic(heuristic_file, copied_heuristics, test_data_list, test_ratio=0.3)
        with open("comparison_test.txt", "a", encoding="utf-8") as f:
            f.write(f"{heuristic_file.__name__}\t{running_steps}\t{average_ms}\t{average_nones}\t{complete_ratio}\t{average_gap}\t{crashed}\n")
            f.close()

def work():
    test_dir = os.path.join("src", "problems", "max_cut", "heuristics", "evolved_heuristics.part2")
    test_data_list = [f"g{i}.mc" for i in [30]]
    test_all_heuristics(test_dir, test_data_list)

work()