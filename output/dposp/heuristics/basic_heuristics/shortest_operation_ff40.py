from src.problems.dposp.components import *

def shortest_operation_ff40(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """Shortest Operation Heuristic for DPOSP.
    
    This heuristic iterates over each production line and attempts to insert the shortest unfulfilled order
    that can be appended to the end of the production line's schedule without violating any constraints.
    
    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "production_line_num" (int): Total number of production lines.
            - "order_num" (int): Total number of orders.
            - "order_quantity" (list[int]): Quantity required for each order.
            - "transition_time" (numpy.array): 3D array of transition time between products on each production line.
            - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): Current scheduling solution.
            - "unfulfilled_orders" (list[int]): List of unfulfilled orders.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check if a production schedule is valid.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This heuristic does not use algorithm_data.
        get_state_data_function (callable): Function that returns the state dictionary for new solution. Not used in this heuristic.
    
    Returns:
        InsertOperator: The operator to insert the selected order into the selected production line.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from state_data
    current_solution = state_data["current_solution"]
    unfulfilled_orders = state_data["unfulfilled_orders"]
    feasible_orders_to_fulfill = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]

    # Extract necessary data from global_data
    production_line_num = global_data["production_line_num"]
    order_quantity = global_data["order_quantity"]
    transition_time = global_data["transition_time"]
    production_rate = global_data["production_rate"]

    # Check if there are any feasible orders to fulfill
    if not feasible_orders_to_fulfill:
        return None, {}

    # Sort unfulfilled orders based on their quantity (shortest first)
    sorted_orders = sorted(feasible_orders_to_fulfill, key=lambda order: order_quantity[order])

    # Iterate over each production line
    for line_id in range(production_line_num):
        # Iterate over the sorted unfulfilled orders
        for order_id in sorted_orders:
            # Check if the production line can produce the product
            product_id = global_data["order_product"][order_id]
            if production_rate[line_id][product_id] == 0:
                continue

            # Generate a new schedule by appending the order to the selected production line
            new_schedule = current_solution.production_schedule[line_id][:]
            new_schedule.append(order_id)

            # Validate the new schedule
            if validation_single_production_schedule(line_id, new_schedule):
                # If valid, create and return the InsertOperator
                return InsertOperator(production_line_id=line_id, order_id=order_id, position=len(new_schedule) - 1), {}

    # If no valid operation is found, return None
    return None, {}