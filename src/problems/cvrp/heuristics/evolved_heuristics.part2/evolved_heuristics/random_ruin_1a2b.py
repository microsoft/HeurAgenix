from src.problems.cvrp.components import BatchRemoveOperator
import random

def random_ruin_1a2b(problem_state: dict, algorithm_data: dict, removal_fraction: float = 0.2, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Random ruin operator that removes a specified fraction of randomly selected customer nodes
    from the current solution. This operator does not violate any capacity constraints as it
    only removes nodes and decreases vehicle loads. The depot node is never removed.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. Must include:
            - "current_solution" (Solution): The current solutions containing routes.
            - "depot" (int): The index of the depot node.
        algorithm_data (dict): Algorithm-specific data.
        removal_fraction (float): The fraction of currently visited customer nodes to remove.
            Default is 0.2 (20%).
            
    Returns:
        (BatchRemoveOperator, dict): The operator to batch remove the selected nodes, 
        and an empty dictionary as the heuristic does not explicitly update algorithm_data.
    """
    current_solution = problem_state["current_solution"]
    depot = problem_state["depot"]
    
    # Collect all visited customer nodes
    visited_customers = []
    for route in current_solution.routes:
        for node in route:
            if node != depot:
                visited_customers.append(node)
                
    if not visited_customers:
        return BatchRemoveOperator(nodes=[]), {}
        
    # Determine the number of nodes to remove
    num_to_remove = max(1, int(len(visited_customers) * removal_fraction))
    
    # Randomly select nodes to remove
    nodes_to_remove = random.sample(visited_customers, min(num_to_remove, len(visited_customers)))
    
    return BatchRemoveOperator(nodes=nodes_to_remove), {}
