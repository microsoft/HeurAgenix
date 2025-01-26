from src.problems.max_cut.components import Solution, InsertNodeOperator
import random

def random_5c59(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """Random node insertion heuristic for Max Cut.

    Args:
        global_data (dict): Contains the global data about the graph.
            - "node_num" (int): The total number of vertices in the graph.
            
        state_data (dict): Contains the current state information.
            - "current_solution" (Solution): The current solution instance.
            - "unselected_count" (int): The number of unselected nodes.
    
    Returns:
        InsertNodeOperator: The operator to insert a node into set A or B.
        dict: Empty dictionary as no algorithm data is updated.
    """
    node_num = global_data['node_num']
    current_solution = state_data['current_solution']
    unselected_nodes = state_data['unselected_nodes']

    # If there are no unselected nodes left, return None.
    if not unselected_nodes:
        return None, {}

    # Randomly choose an unselected node.
    node_to_insert = random.choice(list(unselected_nodes))
    
    # Randomly decide to which set the node will be inserted.
    target_set = random.choice(['A', 'B'])

    # Create the operator.
    operator = InsertNodeOperator(node=node_to_insert, target_set=target_set)
    
    # Return the operator and an empty algorithm data dictionary.
    return operator, {}