from src.problems.dposp.components import *
import numpy as np

def greedy_by_order_density_de77(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """Heuristic to prioritize orders based on a weighted combination of deadline tightness, production feasibility, and potential flexibility for future scheduling.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production speeds for each product on each production line.
            - transition_time (numpy.array): 3D array of transition times between products for each production line.
            - order_deadline (numpy.array): 1D array of deadlines for each order.
            - order_quantity (numpy.array): 1D array of quantities required for each order.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - production_line_num (int): Total number of production lines.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current scheduling solution.
            - feasible_orders_to_fulfill (list[int]): List of feasible orders that can be fulfilled.
            - validation_single_production_schedule (callable): Function to validate a single production schedule.
            - get_time_cost_delta (callable): Function to compute the time cost delta for inserting an order.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not modify algorithm data.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the origin solution.
        kwargs: Hyper-parameters for the heuristic:
            - flexibility_weight (float, default=1.0): Weight for the flexibility factor in the scoring system.

    Returns:
        AppendOperator: The operator to append the selected order to a production line's schedule.
        dict: Updated algorithm data (empty in this case).
    """
    # Extract necessary data from global_data and state_data
    production_rate = global_data["production_rate"]
    transition_time = global_data["transition_time"]
    order_deadline = global_data["order_deadline"]
    order_quantity = global_data["order_quantity"]
    order_product = global_data["order_product"]
    production_line_num = global_data["production_line_num"]

    current_solution = state_data["current_solution"]
    feasible_orders_to_fulfill = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]
    get_time_cost_delta = state_data["get_time_cost_delta"]

    # Hyper-parameters
    flexibility_weight = kwargs.get("flexibility_weight", 1.0)

    # Return None if no feasible orders are available
    if not feasible_orders_to_fulfill:
        return None, {}

    # Step 1: Calculate scores for feasible orders
    best_operator = None
    best_score = -float('inf')

    for order_id in feasible_orders_to_fulfill:
        product_id = order_product[order_id]
        deadline_weight = 1 / (order_deadline[order_id] + 1)
        quantity_penalty = 1 / order_quantity[order_id]

        for line_id in range(production_line_num):
            # Skip if the production line cannot produce this product
            if production_rate[line_id][product_id] == 0:
                continue

            current_schedule = current_solution.production_schedule[line_id]

            # Simulate appending the order at the end of the production line
            new_schedule = current_schedule + [order_id]

            # Validate the new schedule
            if not validation_single_production_schedule(line_id, new_schedule):
                continue

            # Calculate flexibility factor based on the number of feasible orders after this insertion
            future_feasibility = sum(
                1 for future_order in feasible_orders_to_fulfill
                if future_order != order_id and production_rate[line_id][order_product[future_order]] > 0
            )
            flexibility_factor = 1 / (future_feasibility + 1)

            # Calculate insertion cost
            delta_time_cost = get_time_cost_delta(line_id, order_id, len(current_schedule))

            # Calculate overall score for this insertion
            score = deadline_weight * quantity_penalty * (flexibility_factor ** flexibility_weight) - delta_time_cost

            # Update the best operator if the score is higher
            if score > best_score:
                best_score = score
                best_operator = AppendOperator(production_line_id=line_id, order_id=order_id)

    # Return the best operator and empty updated algorithm data
    if best_operator:
        return best_operator, {}
    return None, {}