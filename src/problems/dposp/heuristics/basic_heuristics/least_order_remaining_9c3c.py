from src.problems.dposp.components import *

def least_order_remaining_9c3c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Heuristic for selecting the next order to append to the production schedule based on least cumulative remaining work.
    The heuristic identifies the order with the shortest processing time left from the unfulfilled orders list and appends it to a production line.

    Args:
        problem_state (dict): The dictionary contains the problem state.

    Returns:
        AppendOperator: Operator that appends the chosen order to the production line's schedule.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from problem_state
    production_rate = problem_state["production_rate"]
    order_quantity = problem_state["order_quantity"]
    feasible_orders_to_fulfill = problem_state["feasible_orders_to_fulfill"]
    production_schedule = problem_state["current_solution"].production_schedule
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]

    # Find the line with the least cumulative remaining work
    min_remaining_work = float('inf')
    target_line_id = None
    for line_id, line_schedule in enumerate(production_schedule):
        remaining_work = sum(order_quantity[order_id] / production_rate[line_id, problem_state["order_product"][order_id]]
                             for order_id in line_schedule if production_rate[line_id, problem_state["order_product"][order_id]] > 0)
        if remaining_work < min_remaining_work:
            min_remaining_work = remaining_work
            target_line_id = line_id

    # If no feasible production line found, return None
    if target_line_id is None:
        return None, {}

    # Find the order with the shortest processing time from the feasible orders list
    min_processing_time = float('inf')
    chosen_order_id = None
    for order_id in feasible_orders_to_fulfill:
        # Check if the production line can produce this product
        if production_rate[target_line_id, problem_state["order_product"][order_id]] > 0:
            processing_time = order_quantity[order_id] / production_rate[target_line_id, problem_state["order_product"][order_id]]
            if processing_time < min_processing_time:
                # Check if appending this order is valid
                new_schedule = production_schedule[target_line_id] + [order_id]
                if validation_single_production_schedule(target_line_id, new_schedule):
                    min_processing_time = processing_time
                    chosen_order_id = order_id

    # If no order can be appended without violating constraints, return None
    if chosen_order_id is None:
        return None, {}

    # Create and return the AppendOperator for the chosen order
    return AppendOperator(target_line_id, chosen_order_id), {}