from src.problems.dposp.components import *

def nearest_order_scheduling_1a5e(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Single-step constructive extension via first-feasible append. Select initial_order = feasible_orders_to_fulfill[0] and attempt to append it to the end of each production line in order. Capability filter: production_rate[line_id][order_product[initial_order]] > 0. Feasibility is enforced exclusively through validation_single_production_schedule on the tentative line schedule (transition constraints and deadlines are handled implicitly by the validator). Acceptance policy: first-improvement/first-feasible; the first line passing validation is chosen, no scoring or tie-breaking beyond iteration order.

    Args:
        problem_state (dict): Contains static information for the DPOSP. Relevant keys:
            - "transition_time" (numpy.array): 3D array of transition time between products on each production line.
            - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
            - "order_deadline" (numpy.array): 1D array of the deadline for each order.
            - "order_product" (numpy.array): 1D array mapping each order to its required product.
            - "current_solution" (Solution): Current scheduling solution.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check whether a production schedule is valid.

        algorithm_data (dict): Contains data necessary for the heuristic, which may be updated and returned.
        **kwargs: Optional hyperparameters for the heuristic, which can be used to tweak its behavior.

    Returns:
        AppendOperator: Operator that appends the selected order to the production schedule.
        dict: Updated algorithm_data containing any new information for future iterations.
    """

    # Unpack necessary data from problem_state
    transition_time = problem_state["transition_time"]
    production_rate = problem_state["production_rate"]
    order_deadline = problem_state["order_deadline"]
    order_product = problem_state["order_product"]
    current_solution = problem_state["current_solution"]
    feasible_orders = problem_state["feasible_orders_to_fulfill"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]

    # Start heuristic logic
    if not feasible_orders:
        # No feasible orders to schedule, return None
        return None, {}

    # Select initial order - either the first feasible order or based on a specific criterion
    initial_order = feasible_orders[0]

    # Find a production line that can produce the initial order
    for line_id, line_schedule in enumerate(current_solution.production_schedule):
        # Check if the production line can produce the product and if the transition is allowed
        product = order_product[initial_order]
        if production_rate[line_id][product] > 0:
            # Found a valid line, try to insert the order
            new_schedule = line_schedule[:] + [initial_order]
            if validation_single_production_schedule(line_id, new_schedule):
                # Valid schedule found, return the corresponding operator
                return AppendOperator(production_line_id=line_id, order_id=initial_order), {}

    # If no valid line is found for the initial order, return None
    return None, {}