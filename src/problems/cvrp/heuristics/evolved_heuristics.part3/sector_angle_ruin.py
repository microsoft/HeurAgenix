from src.problems.cvrp.components import BatchRemoveOperator, Solution
import numpy as np
import random

def sector_angle_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    空间扇区毁灭：以质心为圆心，随机选取一个角度区间，移除该扇区内所有节点。
    """
    current_solution = problem_state["current_solution"]
    node_num = problem_state["node_num"]
    depot = problem_state["depot"]
    coords = problem_state.get("coordinates")
    if coords is None:
        return BatchRemoveOperator(nodes=[]), algorithm_data
    coords = np.array(coords)
    # 计算质心
    center = coords.mean(axis=0)
    # 随机选取扇区角度
    angle = random.uniform(0, 2 * np.pi)
    width = kwargs.get("angle_width", np.pi / 3)  # 60度扇区
    # 计算每个点相对质心的极角
    rel = coords - center
    thetas = np.arctan2(rel[:,1], rel[:,0])
    # 选取扇区内的点
    mask = ((thetas >= angle) & (thetas <= angle + width)) | ((thetas + 2*np.pi >= angle) & (thetas + 2*np.pi <= angle + width))
    nodes_to_remove = [i for i in range(node_num) if mask[i] and i != depot]
    # 若数量太少，扩大扇区
    if len(nodes_to_remove) < max(10, node_num // 20):
        width = np.pi / 2
        mask = ((thetas >= angle) & (thetas <= angle + width)) | ((thetas + 2*np.pi >= angle) & (thetas + 2*np.pi <= angle + width))
        nodes_to_remove = [i for i in range(node_num) if mask[i] and i != depot]
    return BatchRemoveOperator(nodes=nodes_to_remove), algorithm_data
