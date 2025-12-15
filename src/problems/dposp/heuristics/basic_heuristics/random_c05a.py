from src.problems.dposp.components import *
import random

def random_c05a(problem_state: dict, algorithm_data: dict, max_attempts: int = 100) -> tuple[AppendOperator, dict]:
    """
    Stochastic constructive append restricted to end-of-line positions. At each iteration, uniformly sample an order from the prefiltered feasible_orders_to_fulfill and a production line, then accept the first append that passes validation_single_production_schedule for that line. Selection policy: first-improvement via random trials (no scoring or ranking), capped by max_attempts; no insertion into middle positions and no cross-line coordination beyond the feasibility filter. Feasibility is enforced line-locally, implicitly respecting production rates, product transitions, and deadlines; global uniqueness and non-delay constraints are assumed embedded in feasible_orders_to_fulfill.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "production_line_num" (int): Total number of production lines.
            - "order_num" (int): Total number of orders.
            - "current_solution" (Solution): Current scheduling solution.
            - "unfulfilled_orders" (list[int]): List of unfulfilled orders.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check if a production schedule is valid.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This heuristic does not use algorithm_data.
        problem_state["get_problem_state"] (callable): Function that returns the state dictionary for new solution. Not used in this heuristic.
        max_attempts (int): Maximum number of attempts to find a valid operation. Defaults to 100.

    Returns:
        AppendOperator: The operator to append the selected order to the selected production line.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from problem_state
    current_solution = problem_state["current_solution"]
    feasible_orders_to_fulfill = problem_state["feasible_orders_to_fulfill"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]

    # Check if there are any feasible orders to fulfill
    if not feasible_orders_to_fulfill:
        return None, {}

    # Attempt to find a valid order to append to a production line
    for _ in range(max_attempts):
        # Randomly select an order from the feasible list
        order_id = random.choice(feasible_orders_to_fulfill)
        # Randomly select a production line
        production_line_id = random.randrange(problem_state["production_line_num"])
        # Generate a new schedule by appending the order to the selected production line
        new_schedule = current_solution.production_schedule[production_line_id][:]
        new_schedule.append(order_id)
        # Validate the new schedule
        if validation_single_production_schedule(production_line_id, new_schedule):
            # If valid, create and return the AppendOperator
            return AppendOperator(production_line_id=production_line_id, order_id=order_id), {}
    
    # If no valid operation is found after max_attempts, return None
    return None, {}