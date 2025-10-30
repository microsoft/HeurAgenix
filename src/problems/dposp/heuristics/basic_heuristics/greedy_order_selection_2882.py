from src.problems.dposp.components import Solution, AppendOperator, InsertOperator

def greedy_order_selection_2882(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy earliest-deadline-first with limited neighborhood and first-feasible acceptance. Unfulfilled orders are sorted by ascending deadline; for each order, only the three least-loaded production lines (by total_time_cost_per_production_line) are considered. Feasibility is screened by production capability (production_rate[line][product] > 0) and then validated via validation_single_production_schedule on the single-line schedule (transition constraints and deadlines enforced implicitly by the validator). Candidate insertion positions per chosen line are restricted to {start, middle, end}; no explicit time-delta computation (get_time_cost_delta not used) and no global revalidation. Selection policy: first-feasible; the algorithm returns the first valid InsertOperator encountered in the order → line (least-loaded first) → position (start, then middle, then end) iteration, without best-improvement scanning. 
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
            - current_solution (Solution): Current scheduling solution.
            - unfulfilled_orders (list[int]): List of unfulfilled orders.
            - total_time_cost_per_production_line (numpy.array): 1D array of the sum of production and transition times for each production line.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.

    Returns:
        InsertOperator: The operator that adds an order to the production line's schedule.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    order_deadline = problem_state['order_deadline']
    order_quantity = problem_state['order_quantity']
    order_product = problem_state['order_product']
    production_rate = problem_state['production_rate']
    transition_time = problem_state['transition_time']
    
    unfulfilled_orders = problem_state['unfulfilled_orders']
    current_solution = problem_state['current_solution']
    total_time_cost_per_production_line = problem_state['total_time_cost_per_production_line']
    validation_single_production_schedule = problem_state['validation_single_production_schedule']
    
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