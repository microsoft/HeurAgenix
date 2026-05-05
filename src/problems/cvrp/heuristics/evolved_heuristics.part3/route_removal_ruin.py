from src.problems.cvrp.components import BatchRemoveOperator
import random

def route_removal_ruin(problem_state: dict, algorithm_data: dict, removal_fraction: float = 0.2, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Route removal ruin operator that entirely removes one or more complete routes.
    """
    current_solution = problem_state["current_solution"]
    routes = current_solution.routes
    depot = problem_state.get("depot", 0)
    
    if not routes:
        return BatchRemoveOperator(nodes=[]), {}
        
    num_routes_to_remove = max(1, int(len(routes) * removal_fraction))
    num_routes_to_remove = min(num_routes_to_remove, len(routes))
    
    routes_to_remove = random.sample(routes, num_routes_to_remove)
    
    nodes_to_remove = []
    for route in routes_to_remove:
        for node in route:
            if node != depot:
                nodes_to_remove.append(node)
                
    return BatchRemoveOperator(nodes=nodes_to_remove), {}