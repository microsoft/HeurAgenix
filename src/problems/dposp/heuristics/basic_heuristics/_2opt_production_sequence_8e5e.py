from src.problems.dposp.components import *
import numpy as np

def _2opt_production_sequence_8e5e(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    Production-line path 2-opt with feasibility screening and best-improvement selection. For each line, it evaluates reversing the contiguous segment (i+1..j) for all i<j, using a transition-only delta: original t(A→B) + t(B→C) versus new t(A→C) + t(C→B), where A=schedule[i], B=schedule[i+1], C=schedule[j]; if j is the last position, the far-end term is omitted (open-chain model). Production rates and deadlines do not enter the scoring; they are enforced by validating the mutated line schedule. Among all lines and pairs, it selects the globally best feasible move (most negative delta) and returns a single ReverseSegmentOperator for that segment. Operates strictly within one production line, respects forbidden transitions via validation, and targets transition-time reduction as a proxy to improve deadline feasibility. Time complexity: O(∑|line|²); constant extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - order_quantity (numpy.array): 1D array of the quantity required for each order.
            - order_deadline (numpy.array): 1D array of the deadline for each order.
            - current_solution (Solution): Current scheduling solution.
            - validation_single_production_schedule (callable): Function to check whether the production schedule is valid.

    Returns:
        ReverseSegmentOperator: The operator that reverse two nodes in the solution to achieve a shorter production schedule.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    production_rate = problem_state["production_rate"]
    transition_time = problem_state["transition_time"]
    order_product = problem_state["order_product"]
    order_quantity = problem_state["order_quantity"]
    order_deadline = problem_state["order_deadline"]
    
    current_solution = problem_state["current_solution"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]
    
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