from src.problems.cvrp.components import RemoveNodesOperator
import random

def sisr_ruin(problem_state: dict, **kwargs) -> tuple:
    """
    SISR Ruin Operator:
    Selects a 'string' of consecutive customers from a route and completely removes them.
    """
    solution = problem_state["current_solution"]
    routes = solution.routes
    valid_routes = [i for i, r in enumerate(routes) if len(r) > 0]
    if not valid_routes:
        return RemoveNodesOperator(nodes=[]), getattr(kwargs, 'info', {})

    target_vehicle = random.choice(valid_routes)
    target_route = routes[target_vehicle]
    string_length = min(len(target_route), random.randint(1, 5))
    start_idx = random.randint(0, max(0, len(target_route) - string_length))
    
    nodes_to_remove = target_route[start_idx : start_idx + string_length]
    
    return RemoveNodesOperator(nodes=nodes_to_remove), kwargs.get('info', {})
