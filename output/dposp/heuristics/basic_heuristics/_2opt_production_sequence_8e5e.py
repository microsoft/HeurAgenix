from src.problems.dposp.components import *
import numpy as np

def _2opt_production_sequence_8e5e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """The 2-opt Production Sequence heuristic adapts the 2-opt approach from TSP to the context of DPOSP. 
    It aims to improve the sequence of production orders on each production line by swapping two non-adjacent orders and re-evaluating the schedule's efficiency.
    By considering the production rates, transition times, and order deadlines, this heuristic explores alternative sequences that could lead to a higher number of completed orders or reduced total production and transition time.
    The heuristic iteratively checks all possible pairs of non-adjacent orders within a production line to determine if a more optimal order sequence can be found, taking care to maintain the feasibility of the schedule with respect to order deadlines.
    If a more efficient sequence is identified, the heuristic generates a ReverseSegmentOperator that applies this improved order sequence to the production line.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current scheduling solution.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.
        get_state_data_function (callable): Function to get the state data for a new solution.

    Returns:
        ReverseSegmentOperator: The operator that reverse two nodes in the solution to achieve a shorter production schedule.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    production_rate = global_data["production_rate"]
    transition_time = global_data["transition_time"]
    order_product = global_data["order_product"]
    order_quantity = global_data["order_quantity"]
    order_deadline = global_data["order_deadline"]
    
    current_solution = state_data["current_solution"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]
    
    best_delta = 0
    best_pair = None
    best_line_id = None
    
    # Iterate over each production line
    for line_id, schedule in enumerate(current_solution.production_schedule):
        # Iterate over all pairs of non-adjacent orders within the production line
        for i in range(len(schedule) - 1):
            for j in range(i + 1, len(schedule)):
                # Calculate the time cost difference if these two orders are swapped
                order_a, order_b = schedule[i], schedule[i + 1]
                order_c, order_d = schedule[j], schedule[j + 1] if j + 1 < len(schedule) else None
                
                product_a, product_b = order_product[order_a], order_product[order_b]
                product_c = order_product[order_c]
                
                transition_time_ab = transition_time[line_id][product_a][product_b]
                transition_time_bc = transition_time[line_id][product_b][product_c] if order_d else 0
                transition_time_ac = transition_time[line_id][product_a][product_c]
                transition_time_cb = transition_time[line_id][product_c][product_b] if order_d else 0
                
                original_time_cost = transition_time_ab + transition_time_bc
                new_time_cost = transition_time_ac + transition_time_cb
                
                delta = new_time_cost - original_time_cost
                
                # Check for an improvement
                if delta < best_delta:
                    new_schedule = schedule[:]
                    new_schedule[i + 1:j + 1] = reversed(new_schedule[i + 1:j + 1])
                    
                    if validation_single_production_schedule(line_id, new_schedule):
                        best_delta = delta
                        best_pair = (i + 1, j)
                        best_line_id = line_id
    
    # If an improvement has been found, create and return the corresponding ReverseSegmentOperator
    if best_pair:
        return ReverseSegmentOperator(best_line_id, [(best_pair[0], best_pair[1])]), {}
    else:
        # No improvement found, return an empty operator
        return None, {}