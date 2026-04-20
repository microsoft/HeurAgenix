from src.problems.cvrp.components import BatchInsertOperator
import math

def regret_insertion_2f3a(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchInsertOperator, dict]:
    """
    Regret-2 Insertion Operator for Recreate Phase.
    For each unvisited node, it considers all feasible insertion positions across all available
    vehicles, calculating the insertion cost (Delta). It then finds the best (C1) and 
    second best (C2) insertion costs. The node with the largest regret (C2 - C1) is inserted first
    into its best position.
    
    This process is repeated in a virtual state (updating routes and loads internally) 
    until all unvisited nodes are placed.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. Must include:
            - "current_solution" (Solution): The current solutions containing routes.
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            - "demands" (numpy.ndarray or list): Demands of nodes.
            - "capacity" (int): Vehicle capacity.
            - "unvisited_nodes" (list[int]): Nodes that need to be inserted.
            - "vehicle_loads" (list[int]): Current load of each vehicle.
            - "depot" (int): The index of the depot node (often 0).
        algorithm_data (dict): Algorithm-specific data.
            
    Returns:
        (BatchInsertOperator, dict): The operator to batch insert the nodes, 
        and an empty dictionary.
    """
    unvisited_nodes = list(problem_state["unvisited_nodes"])
    if not unvisited_nodes:
        return BatchInsertOperator(insertions=[]), {}

    distance_matrix = problem_state["distance_matrix"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    
    # 1. Setup virtual states to track routes and loads without directly mutating the environment yet
    # We deepcopy the basic structures to simulate sequential insertions
    virtual_routes = [list(route) for route in problem_state["current_solution"].routes]
    virtual_loads = list(problem_state["vehicle_loads"])
    
    insertions = []
    
    # Process until all unvisited nodes are inserted (or we completely fail to find feasible spots)
    while unvisited_nodes:
        best_node_to_insert = None
        max_regret = -1.0
        best_insertion_for_node = None # Format: (vehicle_id, position)
        
        # 2. Evaluate all unvisited nodes
        for node in unvisited_nodes:
            demand = demands[node]
            
            # Collect all feasible insertions for this node: list of (cost, vid, pos)
            feasible_insertions = []
            
            for vid, route in enumerate(virtual_routes):
                if virtual_loads[vid] + demand > capacity:
                    continue # Capacity constraint fails, skip this vehicle entirely
                
                # Check all insertion positions in this route
                # Route structure implicit: usually [node1, node2, ...] (depot is implied at start and end)
                # Position pos means inserting before route[pos].
                for pos in range(1, len(route) + 1):
                    prev_node = route[pos - 1] if pos > 0 else depot
                    next_node = route[pos] if pos < len(route) else depot
                    
                    # Calculate Delta Cost
                    delta_cost = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]
                    feasible_insertions.append((delta_cost, vid, pos))
            
            # 3. Calculate Regret-2
            if not feasible_insertions:
                # No feasible spot for this node. It means capacities are completely locked.
                # In standard ALNS, we might keep it unassigned or penalize. We skip it and hope others move.
                continue
                
            if len(feasible_insertions) == 1:
                # Only 1 feasible spot! This is extremely critical, regret is basically infinite.
                regret = float('inf')
                best_ins = feasible_insertions[0]
            else:
                # Sort to find Top 1 and Top 2 costs
                feasible_insertions.sort(key=lambda x: x[0])
                c1 = feasible_insertions[0][0]
                c2 = feasible_insertions[1][0]
                regret = c2 - c1
                best_ins = feasible_insertions[0]
                
            # Keep track of the node with maximum regret
            # We use > to strictly favor larger regret; if a node has inf regret it gets picked ASAP
            if regret > max_regret:
                max_regret = regret
                best_node_to_insert = node
                best_insertion_for_node = (best_ins[1], best_ins[2]) # (vid, pos)
                
        # 4. Apply the best insertion to our virtual state
        if best_node_to_insert is None:
            # We have Unvisited Nodes but NO feasible spots left anywhere! 
            # In complete frameworks, we'd open a new vehicle if allowed, or return a partial assignment.
            # Here, we insert as much as possible, then break.
            break 
            
        vid, pos = best_insertion_for_node
        virtual_routes[vid].insert(pos, best_node_to_insert)
        virtual_loads[vid] += demands[best_node_to_insert]
        unvisited_nodes.remove(best_node_to_insert)
        
        insertions.append((vid, pos, best_node_to_insert))

    return BatchInsertOperator(insertions=insertions), {}
