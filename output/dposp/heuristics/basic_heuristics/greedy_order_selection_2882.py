from src.problems.dposp.components import Solution, AppendOperator, InsertOperator

def greedy_order_selection_2882(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy heuristic algorithm for the DPOSP.
    This algorithm attempts to construct a solution by iteratively adding the most suitable order to a production line's schedule based on a specific criterion such as closest deadline or shortest processing time.

    Args:
        global_data (dict): Contains the global problem data, including:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
        
        state_data (dict): Contains the current state of the solution, including:
            - current_solution (Solution): Current scheduling solution.
            - unfulfilled_orders (list[int]): List of unfulfilled orders.
            - total_time_cost_per_production_line (numpy.array): 1D array of the sum of production and transition times for each production line.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.

    Returns:
        InsertOperator: The operator that adds an order to the production line's schedule.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    order_deadline = global_data['order_deadline']
    order_quantity = global_data['order_quantity']
    order_product = global_data['order_product']
    production_rate = global_data['production_rate']
    transition_time = global_data['transition_time']
    
    unfulfilled_orders = state_data['unfulfilled_orders']
    current_solution = state_data['current_solution']
    total_time_cost_per_production_line = state_data['total_time_cost_per_production_line']
    validation_single_production_schedule = state_data['validation_single_production_schedule']
    
    # Sort unfulfilled orders by closest deadline
    unfulfilled_orders.sort(key=lambda order: order_deadline[order])
    
    for order in unfulfilled_orders:
        product = order_product[order]
        quantity = order_quantity[order]
        deadline = order_deadline[order]
        
        # Evaluate a subset of production lines (e.g., top 3 with the least total time cost)
        line_indices = sorted(range(len(total_time_cost_per_production_line)), key=lambda i: total_time_cost_per_production_line[i])[:3]
        
        for line in line_indices:
            # Check production feasibility
            if production_rate[line][product] == 0:
                continue
            
            # Evaluate a few potential positions (start, middle, end)
            positions = [0, len(current_solution.production_schedule[line]) // 2, len(current_solution.production_schedule[line])]
            
            for position in positions:
                new_schedule = [order[:] for order in current_solution.production_schedule]
                new_schedule[line].insert(position, order)
                
                if validation_single_production_schedule(line, new_schedule[line]):
                    return InsertOperator(line, order, position), {}
    
    # If no valid insertion is found, return None
    return None, {}