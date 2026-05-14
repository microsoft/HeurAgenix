from src.problems.cvrp.components import ReplaceSolutionOperator

def replace_routes(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    A utility heuristic to replace the entire solution with a provided set of routes.
    """
    routes = kwargs.get("routes", [])
    return ReplaceSolutionOperator(routes=routes), algorithm_data
