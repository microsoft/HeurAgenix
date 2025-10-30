from src.problems.mkp.components import *

def greedy_by_profitto_weight_ratio_3aad(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy first-feasible selection by lexicographic multi-resource profit-to-weight ratios. For each item i, construct a ratio vector r_i over all resource dimensions with entries r_i[j] = p_i / w_{j,i}, assigning 0 when w_{j,i} = 0 (zero-weight resources are not treated as infinite utility). Sort items in descending lexicographic order by r_i; ties are resolved by subsequent dimensions implicitly via Python’s list comparison. Traverse the sorted list and pick the first item that fits within the remaining capacities on all resources (resource-major layout: weights[j][i]). Selection policy: first-improvement; the algorithm returns the earliest feasible item in the sorted order rather than the globally best under any aggregate metric. A validity check via get_problem_state is applied before emitting the AddOperator.     
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "sorted_items_by_ratio" (list[int]): Sorted item indices by their profit-to-weight ratio.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "current_solution" (Solution): Current solution instance.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            
        algorithm_data (dict): Contains data specific to the algorithm's execution. Not used in this heuristic.        
        
    Returns:
        AddOperator: Operator to add the selected item to the current solution.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Extract necessary data from problem_state
    weights = problem_state["weights"]
    profits = problem_state["profits"]
    item_num = problem_state["item_num"]
    capacities = problem_state["capacities"]
    current_solution = problem_state["current_solution"]
    remaining_capacity = problem_state["remaining_capacity"]
    
    profit_to_weight_ratio = [
        [p / w if w != 0 else 0 for w in weights_dim]
        for p, weights_dim in zip(profits, zip(*weights))
    ]
    sorted_items_by_ratio = sorted(
        range(item_num),
        key=lambda i: profit_to_weight_ratio[i],
        reverse=True
    )
    # Iterate over sorted items by profit-to-weight ratio
    for item_index in sorted_items_by_ratio:
        # Check if the item is already in the knapsack
        if current_solution.item_inclusion[item_index]:
            continue
        
        # Check if adding the item violates any resource constraints
        item_fits = True
        for resource_index in range(len(capacities)):
            if weights[resource_index][item_index] > remaining_capacity[resource_index]:
                item_fits = False
                break
        
        # If the item fits, return the corresponding AddOperator
        if item_fits:
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_index] = True
            new_problem_state = problem_state["get_problem_state"](Solution(new_solution))
            
            # If the new solution is valid, return the operator and an empty algorithm data dictionary
            if new_problem_state is not None:
                return AddOperator(item_index), {}
    
    # If no item can be added without violating constraints, return None
    return None, {}