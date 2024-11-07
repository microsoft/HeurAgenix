# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features of global data for DPOSP.

    Args:
        global_data (dict): The global data dict containing the global static information data.

    Returns:
        dict: A dictionary of extracted features.
    """
    # Calculate the average production rate across all production lines for each product
    average_production_rate = np.mean(global_data["production_rate"], axis=0)

    # Find the maximum transition time across all transitions
    maximum_transition_time = np.amax(global_data["transition_time"])

    # Calculate the average order quantity across all orders
    average_order_quantity = np.mean(global_data["order_quantity"])

    # Calculate the variance in order deadlines
    deadline_variance = np.var(global_data["order_deadline"])

    # Calculate the ratio of number of products to number of orders
    product_to_order_ratio = global_data["product_num"] / global_data["order_num"]

    # Calculate average deadline per product
    average_deadline_per_product = np.array([
        np.mean(global_data["order_deadline"][global_data["order_product"] == p])
        for p in range(global_data["product_num"])
    ])

    # Calculate the production line utilization potential
    production_rates_for_orders = global_data["production_rate"][:, global_data["order_product"]]
    order_production_times = global_data["order_quantity"] / production_rates_for_orders
    production_line_utilization_potential = np.sum(order_production_times, axis=1) / np.max(global_data["order_deadline"])

    return {
        "average_production_rate": average_production_rate,
        "maximum_transition_time": maximum_transition_time,
        "average_order_quantity": average_order_quantity,
        "deadline_variance": deadline_variance,
        "product_to_order_ratio": product_to_order_ratio,
        "average_deadline_per_product": average_deadline_per_product,
        "production_line_utilization_potential": production_line_utilization_potential
    }

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features of state data for DPOSP.

    Args:
        global_data (dict): The global data dict containing the global static information.
        state_data (dict): The state data dict containing the current dynamic state data.

    Returns:
        dict: A dictionary of extracted features representing the current state.
    """
    # Calculate the ratio of fulfilled orders to total orders
    fulfillment_ratio = state_data["fulfilled_order_num"] / global_data["order_num"]

    # Calculate the average time cost per production line
    average_time_cost = np.mean(state_data["total_time_cost_per_production_line"])

    # Find the maximum time cost across all production lines
    max_time_cost = np.max(state_data["total_time_cost_per_production_line"])

    # Sum the quantities of unfulfilled orders
    unfulfilled_order_quantity_sum = np.sum(global_data["order_quantity"][state_data["unfulfilled_orders"]])

    # Calculate the variance of deadlines for unfulfilled orders
    unfulfilled_order_deadline_variance = np.var(global_data["order_deadline"][state_data["unfulfilled_orders"]])

    # Count the number of feasible orders that can be fulfilled
    feasible_order_count = len(state_data["feasible_orders_to_fulfill"])

    # Check if the current solution is valid
    solution_validity = state_data["validation_solution"](state_data["current_solution"])

    return {
        "fulfillment_ratio": fulfillment_ratio,
        "average_time_cost": average_time_cost,
        "max_time_cost": max_time_cost,
        "unfulfilled_order_quantity_sum": unfulfilled_order_quantity_sum,
        "unfulfilled_order_deadline_variance": unfulfilled_order_deadline_variance,
        "feasible_order_count": feasible_order_count,
        "solution_validity": solution_validity
    }