from src.problems.dposp.components import Solution, InsertOperator

def farthest_deadline_insertion_7e8a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Farthest Deadline Insertion heuristic for DPOSP.
    
    This heuristic selects the unfulfilled order with the farthest deadline and attempts to insert it into a 
    production schedule where it causes the least increase in overall production and transition times, without 
    violating any existing deadlines. The heuristic iterates over each production line, evaluating the insertion 
    of the farthest deadline order at various positions, considering both the production speed of the line for 
    the product and the transition times between orders. The goal is to minimize the impact on the current schedule 
    while ensuring that all orders are completed before their respective deadlines.

    Args:
        global_data (dict): Contains the global static information data.
            - "production_rate" (numpy.array): 2D array of production times for each product on each line.
            - "transition_time" (numpy.array): 3D array of transition times between products on each line.
            - "order_deadline" (numpy.array): 1D array of the deadline for each order.
            - "production_line_num" (int): Total number of production lines.

        state_data (dict): Contains the current dynamic state data.
            - "current_solution" (Solution): Current scheduling solution.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to check if a production schedule is valid.

    Returns:
        InsertOperator: Operator to insert the selected order at the best position found.
        dict: Empty dictionary as no additional algorithm data is updated by this heuristic.
    """
    # Retrieve necessary data from global_data
    order_deadline = global_data['order_deadline']
    production_line_num = global_data['production_line_num']
    production_rate = global_data['production_rate']
    transition_time = global_data['transition_time']
    
    # Retrieve necessary data from state_data
    current_solution = state_data['current_solution']
    feasible_orders_to_fulfill = state_data['feasible_orders_to_fulfill']
    validation_single_production_schedule = state_data['validation_single_production_schedule']
    
    # If no feasible orders to fulfill, return None
    if not feasible_orders_to_fulfill:
        return None, {}
    
    # Select the unfulfilled order with the farthest deadline
    farthest_deadline_order = max(feasible_orders_to_fulfill, key=lambda x: order_deadline[x])
    product_required = global_data['order_product'][farthest_deadline_order]
    
    # Initialize variables for the best insertion
    best_cost_increase = float('inf')
    best_line_id = None
    best_position = None
    
    # Iterate over each production line to find the best insertion position
    for line_id in range(production_line_num):
        # Skip if the production line cannot produce the required product
        if production_rate[line_id][product_required] == 0:
            continue
        
        schedule = current_solution.production_schedule[line_id]
        for position in range(len(schedule) + 1):
            new_schedule = schedule[:]
            new_schedule.insert(position, farthest_deadline_order)
            # Validate the new schedule
            if validation_single_production_schedule(line_id, new_schedule):
                # Calculate the cost increase if the farthest deadline order was inserted here
                # Dummy cost calculation, should be replaced with actual cost estimation logic
                cost_increase = position  # Placeholder for actual cost increase calculation
                
                # Update the best insertion if this is the lowest cost increase found so far
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_line_id = line_id
                    best_position = position
    
    # If no feasible insertion was found, return None
    if best_line_id is None:
        return None, {}
    
    # Create the operator to perform the insertion
    insert_operator = InsertOperator(production_line_id=best_line_id, order_id=farthest_deadline_order, position=best_position)
    
    return insert_operator, {}