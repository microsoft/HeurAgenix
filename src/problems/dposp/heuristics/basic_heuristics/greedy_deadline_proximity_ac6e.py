from src.problems.dposp.components import AppendOperator, InsertOperator, Solution
import numpy as np

def greedy_deadline_proximity_ac6e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """Greedy Deadline Proximity heuristic for DPOSP.
    This heuristic attempts to construct a production schedule by iteratively appending orders to production lines based on their proximity to their deadline and the transition times between orders. 
    It begins with an empty schedule for each production line and selects orders based on the closest deadline from the current time in the schedule. 
    The heuristic also accounts for the transition time from the last scheduled order to the potential new order and production speeds, ensuring that the added order can be completed before its deadline without violating transition constraints. 
    Orders are chosen to maximize the number of orders fulfilled, considering the remaining processing time and capacity constraints of each production line.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current scheduling solution.
            - feasible_orders_to_fulfill (list[int]): The feasible orders that can be fulfilled based on the current solution without delaying other planned orders.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.
        
    Returns:
        InsertOperator: The operator that adds an order to a production line's schedule.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    # Extract required data from global_data and state_data
    production_rate = global_data['production_rate']
    transition_time = global_data['transition_time']
    order_product = global_data['order_product']
    order_quantity = global_data['order_quantity']
    order_deadline = global_data['order_deadline']
    
    current_solution = state_data['current_solution']
    feasible_orders_to_fulfill = state_data['feasible_orders_to_fulfill']
    validation_single_production_schedule = state_data['validation_single_production_schedule']
    
    # Collect all potential (order, line, position) tuples and sort them by deadline proximity
    potential_options = []
    
    for order_id in feasible_orders_to_fulfill:
        product = order_product[order_id]
        quantity = order_quantity[order_id]
        deadline = order_deadline[order_id]
        
        for line_id in range(global_data['production_line_num']):
            if production_rate[line_id][product] > 0:  # Check if the production line can produce this product
                for position in range(len(current_solution.production_schedule[line_id]) + 1):
                    potential_options.append((order_id, line_id, position, deadline))
    
    # Sort options by deadline proximity
    potential_options.sort(key=lambda x: x[3])
    
    # Try to find a valid insertion
    for order_id, line_id, position, deadline in potential_options:
        new_schedule = current_solution.production_schedule[line_id][:]
        new_schedule.insert(position, order_id)
        
        if validation_single_production_schedule(line_id, new_schedule):
            return InsertOperator(line_id, order_id, position), {}
    
    # If no valid insertion is found, return None
    return None, {}