from src.problems.mkp.components import *

def greedy_by_least_remaining_capacity_05f2(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy by Least Remaining Capacity heuristic for the Multidimensional Knapsack Problem.
    This heuristic selects the item that, when added, leaves the least remaining capacity in the knapsack.
    
    Args:
        global_data (dict): Contains global information about the problem instance.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            
        state_data (dict): Contains the current state of the solution.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "feasible_items_to_add" (list[int]): Item indices that can be added without violating constraints.
            
        algorithm_data (dict): Contains data specific to the algorithm's execution. Not used in this heuristic.
        
        get_state_data_function (callable): Function to get the state data for a new solution.
        
        **kwargs: Additional hyperparameters. Not used in this heuristic.

    Returns:
        AddOperator: Operator to add the selected item to the knapsack.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Initialize variables to track the best item and its corresponding remaining capacity
    best_item_index = None
    least_remaining_capacity = float('inf')
    
    # Iterate over all feasible items to find the one that leaves the least remaining capacity
    for item_index in state_data["feasible_items_to_add"]:
        # Calculate the new remaining capacity if the item is added
        new_remaining_capacity = [
            state_data["remaining_capacity"][i] - global_data["weights"][i][item_index]
            for i in range(global_data["resource_num"])
        ]
        
        # Check if the new remaining capacity is valid (all values should be >= 0)
        if all(cap >= 0 for cap in new_remaining_capacity):
            total_remaining_capacity = sum(new_remaining_capacity)
            # Update the best item if this one leaves less remaining capacity
            if total_remaining_capacity < least_remaining_capacity:
                best_item_index = item_index
                least_remaining_capacity = total_remaining_capacity
    
    # If a best item was found, return the corresponding AddOperator and an empty algorithm data dictionary
    if best_item_index is not None:
        return AddOperator(best_item_index), {}
    # If no item can be added without violating constraints, return None
    else:
        return None, {}