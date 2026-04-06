from src.problems.cvrp.components import BatchRemoveOperator
import random
import numpy as np

def radial_ruin_3c4d(problem_state: dict, algorithm_data: dict, removal_fraction: float = 0.2, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Radial ruin operator that removes a spatially clustered set of nodes.
    It randomly selects a 'seed' customer that is currently visited, and then removes it
    along with its closest `num_to_remove - 1` neighbors. This effectively destroys a 
    geographic neighborhood to allow the recreate phase to rebuild it more optimally.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. Must include:
            - "current_solution" (Solution): The current solutions containing routes.
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            - "depot" (int): The index of the depot node.
        algorithm_data (dict): Algorithm-specific data.
        removal_fraction (float): The fraction of currently visited customer nodes to remove.
            Default is 0.2 (20%).
            
    Returns:
        (BatchRemoveOperator, dict): The operator to batch remove the selected nodes, 
        and an empty dictionary as the heuristic does not explicitly update algorithm_data.
    """
    current_solution = problem_state["current_solution"]
    distance_matrix = problem_state["distance_matrix"]
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
    num_to_remove = min(num_to_remove, len(visited_customers))
    
    # Randomly select a seed customer
    seed_node = random.choice(visited_customers)
    
    # Calculate distances from the seed node to all other visited customers
    visited_set = set(visited_customers)
    
    # Sort the visited customers by distance to the seed node
    # x is a node, distance_matrix[seed_node][x] is the distance
    sorted_neighbors = sorted([node for node in visited_customers], key=lambda x: distance_matrix[seed_node][x])
    
    # The first `num_to_remove` nodes in this sorted list (including seed itself at idx 0)
    # will be our radial cluster context.
    nodes_to_remove = sorted_neighbors[:num_to_remove]
    
    return BatchRemoveOperator(nodes=nodes_to_remove), {}
