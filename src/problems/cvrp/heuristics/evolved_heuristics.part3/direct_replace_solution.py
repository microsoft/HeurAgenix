from src.problems.cvrp.components import ReplaceSolutionOperator

def direct_replace_solution(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    Directly replaces the current solution with the provided target routes.
    Passed via kwargs:
        target_routes: list of missing/new routes for the solution.
    """
    target_routes = kwargs.get("target_routes", [])
    return ReplaceSolutionOperator(routes=target_routes), algorithm_data
