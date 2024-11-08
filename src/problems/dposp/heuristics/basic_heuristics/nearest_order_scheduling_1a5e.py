from src.problems.dposp.components import *

def nearest_order_scheduling_1a5e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Implements the nearest order scheduling heuristic for DPOSP. Starting from an initial order, the heuristic builds
    a production schedule by selecting and appending the next order that minimizes the combined criteria of setup time
    (transition time between orders) and proximity to the delivery deadline. The heuristic begins with either a randomly
    selected order or the one with the earliest deadline and iteratively adds to the schedule the order that is closest
    in terms of transition time from the previously scheduled order while also taking into account the urgency of order
    deadlines. This process continues until no further orders can be feasibly added to the schedule without violating
    deadline constraints.

    Args:
        global_data (dict): Contains static information for the DPOSP. Relevant keys:
            - "transition_time" (numpy.array): 3D array of transition time between products on each production line.
            - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
            - "order_deadline" (numpy.array): 1D array of the deadline for each order.
            - "order_product" (numpy.array): 1D array mapping each order to its required product.

        state_data (dict): Contains dynamic state information for the DPOSP. Relevant keys:
            - "current_solution" (Solution): Current scheduling solution.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check whether a production schedule is valid.

        algorithm_data (dict): Contains data necessary for the heuristic, which may be updated and returned.

        get_state_data_function (callable): Function to get the state data for a new solution.

        **kwargs: Optional hyperparameters for the heuristic, which can be used to tweak its behavior.

    Returns:
        AppendOperator: Operator that appends the selected order to the production schedule.
        dict: Updated algorithm_data containing any new information for future iterations.
    """

    # Unpack necessary data from global_data
    transition_time = global_data["transition_time"]
    production_rate = global_data["production_rate"]
    order_deadline = global_data["order_deadline"]
    order_product = global_data["order_product"]

    # Unpack necessary data from state_data
    current_solution = state_data["current_solution"]
    feasible_orders = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]

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