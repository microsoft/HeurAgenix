# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): The global data dict containing the global instance data with:
            - "capacity" (int): The capacity for each bin.
            - "item_num" (int): Total item number.

    Returns:
        The feature of global data, which can represents the global data with:
            - bin_capacity (int): The capacity of each bin.
            - total_items (int): Total number of items.
            - average_item_size (float): The average size of items.
            - item_size_variance (float): The variance of item sizes.
            - min_item_size (int): The size of the smallest item.
            - max_item_size (int): The size of the largest item.
    """
    # Initialize the features dictionary
    features = {}

    # Extract bin capacity from global data
    features['bin_capacity'] = global_data['capacity']

    # Extract total number of items from global data
    features['total_items'] = global_data['item_num']

    # Assuming we have access to item sizes, which we do not currently have
    # Here, a placeholder for item sizes is used; replace it with actual data
    item_sizes = np.random.randint(1, global_data['capacity'], size=global_data['item_num'])

    # Calculate average item size
    features['average_item_size'] = np.mean(item_sizes)

    # Calculate item size variance
    features['item_size_variance'] = np.var(item_sizes)

    # Find minimum item size
    features['min_item_size'] = np.min(item_sizes)

    # Find maximum item size
    features['max_item_size'] = np.max(item_sizes)

    return features

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): The global data dict containing the global instance data with:
            - "capacity" (int): The capacity for each bin.
            - "item_num" (int): Total item number.
        state_data (dict): The state data dict containing the solution state data with:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_item_size" (int): The size of current item to pack.
            - "used_bin_num" (int): The number of bins that has been used.
            - "used_capacity" (list[int]): List of used capacity for each bin.
            - "remaining_capacity" (list[int]): List of remaining capacity for each bin.
            - "num_items_in_box" (int): Total number of packed items.
            - "num_items_not_in_box" (int): Total number of unpacked items.
            - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.

    Returns:
        The feature of current solution, which can represents the current state with:
            - used_bin_num (int): The number of bins currently in use.
            - current_item_size (int): The size of the next item to be packed.
            - average_used_capacity (float): The average capacity used in the bins.
            - average_remaining_capacity (float): The average remaining capacity in the bins.
            - packed_item_ratio (float): Ratio of packed items to total items.
            - max_remaining_capacity (int): The maximum remaining capacity in any bin.
            - min_remaining_capacity (int): The minimum remaining capacity in any bin.
            - solution_validity (bool): Whether the current solution is valid.
    """
    # Initialize the features dictionary
    features = {}

    # Feature: Number of bins currently in use
    features['used_bin_num'] = state_data['used_bin_num']

    # Feature: Size of the current item to be packed
    features['current_item_size'] = state_data['current_item_size']

    # Feature: Average used capacity in bins (handle division by zero)
    if len(state_data['used_capacity']) > 0:
        features['average_used_capacity'] = sum(state_data['used_capacity']) / len(state_data['used_capacity'])
    else:
        features['average_used_capacity'] = 0.0

    # Feature: Average remaining capacity in bins (handle division by zero)
    if len(state_data['remaining_capacity']) > 0:
        features['average_remaining_capacity'] = sum(state_data['remaining_capacity']) / len(state_data['remaining_capacity'])
    else:
        features['average_remaining_capacity'] = 0.0

    # Feature: Ratio of packed items to total items (handle division by zero)
    total_items = state_data['num_items_in_box'] + state_data['num_items_not_in_box']
    if total_items > 0:
        features['packed_item_ratio'] = state_data['num_items_in_box'] / total_items
    else:
        features['packed_item_ratio'] = 0.0

    # Feature: Maximum remaining capacity in any bin (handle empty list)
    if len(state_data['remaining_capacity']) > 0:
        features['max_remaining_capacity'] = max(state_data['remaining_capacity'])
    else:
        features['max_remaining_capacity'] = 0

    # Feature: Minimum remaining capacity in any bin (handle empty list)
    if len(state_data['remaining_capacity']) > 0:
        features['min_remaining_capacity'] = min(state_data['remaining_capacity'])
    else:
        features['min_remaining_capacity'] = 0

    # Feature: Check if the current solution is valid
    features['solution_validity'] = state_data['validation_solution'](state_data['current_solution'])

    return features