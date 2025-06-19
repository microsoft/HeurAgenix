from src.problems.dposp.components import *

def maximum_remaining_work_order_ec9c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Heuristic that selects the order with the most remaining work that can be feasibly scheduled on any production line
    without violating existing order deadlines. The chosen order is then appended or inserted into the most appropriate
    production line's schedule, aiming to maximize the number of orders fulfilled before their respective deadlines.
    
    Args:
        global_data (dict): Contains global static problem information, including:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
        state_data (dict): Contains the current state of the solution, including:
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