# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features from global data for the MKP.

    Args:
        global_data (dict): The global data dict containing:
            - "item_num" (int): Total number of items.
            - "resource_num" (int): Number of resource dimensions.
            - "profits" (numpy.array): Profit values of items.
            - "weights" (numpy.array): Resource consumption of items.
            - "capacities" (numpy.array): Capacities of each resource dimension.

    Returns:
        dict: A dictionary with keys representing feature names and their calculated values.
    """
    features = {}

    # Calculate the average profit of items
    features['average_profit'] = np.mean(global_data["profits"])

    # Calculate the variance of the profit values
    features['profit_variance'] = np.var(global_data["profits"])

    # Calculate the average weight per resource and weight variance per resource
    features['average_weight_per_resource'] = np.mean(global_data["weights"], axis=0).tolist()
    features['weight_variance_per_resource'] = np.var(global_data["weights"], axis=0).tolist()

    # Calculate the capacity to weight ratio for each resource
    total_weights = np.sum(global_data["weights"], axis=1)
    features['capacity_to_weight_ratio'] = global_data["capacities"] / total_weights

    # Calculate the profit to weight ratio for each item
    # Prevent division by zero by adding a small epsilon where weights are zero
    epsilon = 1e-10
    weights_with_epsilon = np.where(np.sum(global_data["weights"], axis=0) == 0, epsilon, np.sum(global_data["weights"], axis=0))
    features['profit_to_weight_ratio'] = global_data["profits"] / weights_with_epsilon

    return features

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features from state data for the MKP.

    Args:
        global_data (dict): Contains global static information data.
        state_data (dict): Contains current dynamic state data.

    Returns:
        dict: A dictionary with keys representing state feature names and their calculated values.
    """
    features = {}

    # Calculate the current solution value (total profit)
    features['current_solution_value'] = state_data['current_profit']

    # Calculate the solution density (ratio of items in the knapsack to total items)
    features['solution_density'] = len(state_data['items_in_knapsack']) / global_data['item_num']

    # Calculate the average remaining capacity across all resources
    features['average_remaining_capacity'] = np.mean(state_data['remaining_capacity'])

    # Calculate the variance of the remaining capacities
    features['remaining_capacity_variance'] = np.var(state_data['remaining_capacity'])

    # Calculate the feasibility ratio (feasible items to add vs total remaining items)
    total_remaining_items = global_data['item_num'] - len(state_data['items_in_knapsack'])
    features['feasibility_ratio'] = len(state_data['feasible_items_to_add']) / total_remaining_items if total_remaining_items > 0 else 0

    # Calculate the utilized capacity ratio for each resource
    features['utilized_capacity_ratio'] = state_data['current_weights'] / global_data['capacities']

    # Calculate the average profit of items included in the current solution
    included_items = np.array(state_data['current_solution'].item_inclusion, dtype=bool)
    included_profits = global_data['profits'][included_items]
    features['item_profitability_in_solution'] = np.mean(included_profits) if included_items.any() else 0

    return features