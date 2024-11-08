from src.problems.dposp.components import *
import random

def random_c05a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, max_attempts: int = 100) -> tuple[AppendOperator, dict]:
    """Random Order Selection Heuristic for DPOSP.
    
    This heuristic randomly selects an unfulfilled order from the list of feasible orders and attempts to append it
    to a randomly selected production line's schedule. The insertion is validated to ensure that it
    maintains a valid schedule respecting all production and transition constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "production_line_num" (int): Total number of production lines.
            - "order_num" (int): Total number of orders.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): Current scheduling solution.
            - "unfulfilled_orders" (list[int]): List of unfulfilled orders.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check if a production schedule is valid.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This heuristic does not use algorithm_data.
        get_state_data_function (callable): Function that returns the state dictionary for new solution. Not used in this heuristic.
        max_attempts (int): Maximum number of attempts to find a valid operation. Defaults to 100.

    Returns:
        AppendOperator: The operator to append the selected order to the selected production line.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from state_data
    current_solution = state_data["current_solution"]
    feasible_orders_to_fulfill = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]

    # Check if there are any feasible orders to fulfill
    if not feasible_orders_to_fulfill:
        return None, {}

    # Attempt to find a valid order to append to a production line
    for _ in range(max_attempts):
        # Randomly select an order from the feasible list
        order_id = random.choice(feasible_orders_to_fulfill)
        # Randomly select a production line
        production_line_id = random.randrange(global_data["production_line_num"])
        # Generate a new schedule by appending the order to the selected production line
        new_schedule = current_solution.production_schedule[production_line_id][:]
        new_schedule.append(order_id)
        # Validate the new schedule
        if validation_single_production_schedule(production_line_id, new_schedule):
            # If valid, create and return the AppendOperator
            return AppendOperator(production_line_id=production_line_id, order_id=order_id), {}
    
    # If no valid operation is found after max_attempts, return None
    return None, {}