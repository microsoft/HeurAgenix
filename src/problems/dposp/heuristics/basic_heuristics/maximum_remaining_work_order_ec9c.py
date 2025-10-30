from src.problems.dposp.components import *

def maximum_remaining_work_order_ec9c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Largest-remaining-work first with first-feasible append. For each feasible order k, compute a remaining-work surrogate Q_k / v_i,P_k for each production line i with positive rate; the per-order score stored uses the last feasible line encountered in iteration, so the ranking can depend on line iteration order. Select the order with the maximal stored remaining-work value.
    Placement policy: scan production lines in index order and validate appending the selected order to the end of the line’s schedule via validation_single_production_schedule; accept the first line that passes. No position search (no insertion), no evaluation of time-cost delta/slack; transitions and deadlines are enforced only through the validator.
    Selection policy: best-improvement in the order space according to the remaining-work score; placement is first-improvement over lines. Acceptance is strict; returns a single AppendOperator for the chosen (line, order).
    Complexity: O(|feasible_orders| · production_line_num) to build scores plus O(production_line_num) for placement; linear extra memory in |feasible_orders|.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
            - current_solution (Solution): Current scheduling solution.
            - feasible_orders_to_fulfill (list): The feasible orders that can be fulfilled based on the current solution.
            - validation_single_production_schedule (callable): Function to check if a production schedule is valid.
    
    Returns:
        The AppendOperator or InsertOperator for the selected order.
        An empty dict as this heuristic does not update the algorithm_data.
    """
    # Calculate the remaining work for each feasible order based on production rate and order quantity
    remaining_work_for_orders = {
        order_id: problem_state['order_quantity'][order_id] / problem_state['production_rate'][prod_line_id, problem_state['order_product'][order_id]]
        for order_id in problem_state['feasible_orders_to_fulfill']
        for prod_line_id in range(problem_state['production_line_num'])
        if problem_state['production_rate'][prod_line_id, problem_state['order_product'][order_id]] > 0
    }

    # Select the order with the maximum remaining work
    max_work_order_id = max(remaining_work_for_orders, key=remaining_work_for_orders.get, default=None)

    # If no order is selected, return None
    if max_work_order_id is None:
        return None, {}

    # Find a production line where the order can be feasibly scheduled
    for prod_line_id in range(problem_state['production_line_num']):
        if problem_state['production_rate'][prod_line_id, problem_state['order_product'][max_work_order_id]] > 0:
            # Check if appending the order is valid
            validation_function = problem_state['validation_single_production_schedule']
            new_schedule = problem_state['current_solution'].production_schedule[prod_line_id] + [max_work_order_id]
            if validation_function(prod_line_id, new_schedule):
                # Return the AppendOperator for the selected order and production line
                return AppendOperator(prod_line_id, max_work_order_id), {}

    # If no valid production line is found, return None
    return None, {}