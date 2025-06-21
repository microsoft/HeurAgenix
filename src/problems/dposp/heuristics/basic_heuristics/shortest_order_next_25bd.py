from src.problems.dposp.components import *

def shortest_order_next_25bd(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """Shortest Order Next heuristic for DPOSP selects the unfulfilled order with the shortest processing time and inserts it into the best position in the production schedule.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - current_solution (Solution): The current solution state.
            - unfulfilled_orders (list[int]): List of unfulfilled orders.
            - feasible_orders_to_fulfill (list): The feasible orders that can be fulfilled based on the current solution.
            - validation_single_production_schedule (callable): Function to check if a production schedule is valid.
            
        algorithm_data (dict): Contains data necessary for this heuristic's state, not used in this heuristic.

    Returns:
        InsertOperator: The operator to insert the selected order into the best position.
        dict: An empty dictionary as this heuristic does not update algorithm data.
    """

    # Extract necessary information
    production_rate = problem_state["production_rate"]
    order_product = problem_state["order_product"]
    order_quantity = problem_state["order_quantity"]
    feasible_orders = problem_state["feasible_orders_to_fulfill"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]
    current_solution = problem_state["current_solution"]
    
    # Check if there are any feasible orders to fulfill
    if not feasible_orders:
        return None, {}

    # Initialize the best order and its best position
    best_order = None
    best_position = None
    best_production_line = None
    shortest_time = float('inf')
    
    # Check if there is feasible order
    if len(feasible_orders) == 0:
        return None, {}

    order_id = feasible_orders[0]
    product_id = order_product[order_id]
    quantity = order_quantity[order_id]
    
    for line_id, production_line in enumerate(current_solution.production_schedule):
        production_time = production_rate[line_id][product_id]
        
        # Skip if production line cannot produce this product
        if production_time == 0:
            continue
        
        # Calculate processing time for this order
        time_to_produce = quantity / production_time
        
        # Check for each possible position in the production line
        for position in range(len(production_line) + 1):
            new_schedule = production_line[:]
            new_schedule.insert(position, order_id)
            
            # Validate the new schedule
            if validation_single_production_schedule(line_id, new_schedule):
                if time_to_produce < shortest_time:
                    best_order = order_id
                    best_position = position
                    best_production_line = line_id
                    shortest_time = time_to_produce

    # If no feasible order is found, return None
    if best_order is None:
        return None, {}

    # Create and return the operator with the best order and its position
    return InsertOperator(production_line_id=best_production_line, order_id=best_order, position=best_position), {}