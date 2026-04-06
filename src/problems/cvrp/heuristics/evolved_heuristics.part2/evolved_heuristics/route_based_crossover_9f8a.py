from src.problems.cvrp.components import ReplaceSolutionOperator
import random

def route_based_crossover_9f8a(problem_state: dict, algorithm_data: dict, target_solution=None, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    Route-Based Crossover (RBX) for CVRP.
    This operator requires two parent solutions. It extracts a random whole route from
    Parent A (current_solution), and uses Parent B (target_solution) for the remaining routes.
    To avoid duplicating nodes, any nodes present in the chosen route from Parent A
    are strictly removed from Parent B's routes.
    Because removing nodes from Valid CVRP routes can only DECREASE capacity loads, 
    the resulting offspring is naturally feasible in terms of capacity constraints!
    
    Args:
        problem_state (dict): The dictionary contains the problem state. Must include:
            - "current_solution" (Solution): The first parent.
            - "depot" (int): The index of the depot node.
        algorithm_data (dict): Algorithm-specific data.
        target_solution (Solution): The second parent solution (e.g. from the Elite Pool).
            If not provided via kwargs, it will attempt to find it in algorithm_data.
            
    Returns:
        (ReplaceSolutionOperator, dict): The operator to wholesale replace the state with the 
        new offspring, and an empty dictionary.
    """
    parent_a = problem_state["current_solution"]
    depot = problem_state["depot"]
    
    if target_solution is None:
        target_solution = algorithm_data.get("target_solution")
    if target_solution is None:
        # If no target solution is provided, fallback to completely ignoring the crossover
        # and returning the current solution unchanged. Or raise an error. We will just return the same.
        return ReplaceSolutionOperator(routes=[list(r) for r in parent_a.routes]), {}

    # 1. Select a random non-empty route from Parent A
    valid_routes_a = [r for r in parent_a.routes if len([n for n in r if n != depot]) > 0]
    if not valid_routes_a:
        return ReplaceSolutionOperator(routes=[list(r) for r in target_solution.routes]), {}
        
    chosen_route_a = random.choice(valid_routes_a)
    
    # 2. Extract the customer nodes bounded by the chosen route
    nodes_in_a = set(n for n in chosen_route_a if n != depot)
    
    offspring_routes = []
    
    # Directly append the chosen route into our offspring
    offspring_routes.append(list(chosen_route_a))
    
    # 3. Inherit everything else from Parent B, but skip nodes that are already in chosen_route_a
    for route_b in target_solution.routes:
        new_route_b = []
        for node in route_b:
            if node == depot:
                new_route_b.append(node)
                continue
            if node not in nodes_in_a:
                new_route_b.append(node)
                
        # Only add the route if it actually visits at least one customer
        if any(n != depot for n in new_route_b):
            offspring_routes.append(new_route_b)
            
    # Pad with empty routes if necessary
    vehicle_num = problem_state.get("vehicle_num", len(parent_a.routes))
    while len(offspring_routes) < vehicle_num:
        offspring_routes.append([depot])
        
    # Strictly truncate if we somehow exceeded vehicle_num (should not happen in this logic, but safe)
    if len(offspring_routes) > vehicle_num:
        offspring_routes = offspring_routes[:vehicle_num]
            
    return ReplaceSolutionOperator(routes=offspring_routes), {}
