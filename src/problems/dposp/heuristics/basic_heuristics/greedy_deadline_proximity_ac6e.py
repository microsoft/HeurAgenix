from src.problems.dposp.components import AppendOperator, InsertOperator, Solution
import numpy as np

def greedy_deadline_proximity_ac6e(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Earliest-deadline-first constructive insertion with first-feasible acceptance. The neighborhood enumerates all (order, production_line, position) triples for orders in feasible_orders_to_fulfill on lines that can produce the order’s product (production_rate[line][product] > 0), across all insertion positions 0..|line_schedule|. Candidates are globally sorted by order_deadline ascending; transition times, production speeds, and deadline feasibility are not scored explicitly—feasibility is delegated entirely to validation_single_production_schedule for the affected line.
    Selection policy: first-improvement. The algorithm returns the first candidate in deadline order whose single-line schedule becomes valid after insertion; no best-improvement search over the full neighborhood, no tie-breaking beyond sort order.
    Neighborhood size and complexity: N = sum_l (|schedule_l| + 1) · |feasible_orders_to_fulfill| candidates; sorting O(N log N), validation up to O(N) calls in the worst case; O(N) extra memory. Tailored to DPOSP through validator-based enforcement of start/end times and forbidden transitions.


    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
            - current_solution (Solution): Current scheduling solution.
            - feasible_orders_to_fulfill (list[int]): The feasible orders that can be fulfilled based on the current solution without delaying other planned orders.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.
        
    Returns:
        InsertOperator: The operator that adds an order to a production line's schedule.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    # Extract required data from problem_state
    production_rate = problem_state['production_rate']
    transition_time = problem_state['transition_time']
    order_product = problem_state['order_product']
    order_quantity = problem_state['order_quantity']
    order_deadline = problem_state['order_deadline']
    
    current_solution = problem_state['current_solution']
    feasible_orders_to_fulfill = problem_state['feasible_orders_to_fulfill']
    validation_single_production_schedule = problem_state['validation_single_production_schedule']
    
    # Collect all potential (order, line, position) tuples and sort them by deadline proximity
    potential_options = []
    
    for order_id in feasible_orders_to_fulfill:
        product = order_product[order_id]
        quantity = order_quantity[order_id]
        deadline = order_deadline[order_id]
        
        for line_id in range(problem_state['production_line_num']):
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