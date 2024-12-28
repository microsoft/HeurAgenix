from src.problems.cvrp.components import *
import numpy as np

def farthest_insertion_6308(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Farthest Insertion Heuristic Algorithm with Improvements for the Capacitated Vehicle Routing Problem (CVRP).

    This heuristic algorithm selects unvisited nodes based on a scoring mechanism that considers insertion cost, route compactness, and vehicle capacity utilization. It periodically performs greedy improvements and inter-vehicle swaps to enhance the solution.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): 2D array representing the distances between nodes.
            - demands (numpy.ndarray): Array of demands for each node.
            - capacity (int): Capacity of each vehicle.
            - depot (int): Index of the depot node.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current routes for all vehicles.
            - unvisited_nodes (list[int]): Nodes not yet visited by any vehicle.
            - last_visited (list[int or None]): Last visited node for each vehicle.
            - vehicle_remaining_capacity (list[int]): Remaining capacity for each vehicle.
        algorithm_data (dict): Algorithm-specific data, updated with any new information during execution.
        get_state_data_function (callable): Function to evaluate the state of a new solution without modifying the original state.
        kwargs: Hyper-parameters for the heuristic algorithm.
            - spread_penalty_weight (float, default=1.0): Weight for penalizing spread in the scoring mechanism.
            - capacity_utilization_weight (float, default=1.0): Weight for vehicle capacity utilization in the scoring mechanism.
            - apply_greedy_frequency (int, default=3): Frequency of performing greedy improvements.
            - apply_swap_frequency (int, default=5): Frequency of performing inter-vehicle swaps.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Updated algorithm data.
    """
    # Extract required data
    distance_matrix = global_data["distance_matrix"]
    demands = global_data["demands"]
    capacity = global_data["capacity"]
    depot = global_data["depot"]

    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data["last_visited"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"]

    # Hyper-parameters
    spread_penalty_weight = kwargs.get("spread_penalty_weight", 1.0)
    capacity_utilization_weight = kwargs.get("capacity_utilization_weight", 1.0)
    apply_greedy_frequency = kwargs.get("apply_greedy_frequency", 3)
    apply_swap_frequency = kwargs.get("apply_swap_frequency", 5)

    # Check if there are no unvisited nodes
    if not unvisited_nodes:
        return None, {}

    # Step 1: Scoring mechanism to prioritize unvisited nodes
    best_score = float('inf')
    best_node = None
    best_vehicle = None
    best_position = None

    for node in unvisited_nodes:
        for vehicle_id, remaining_capacity in enumerate(vehicle_remaining_capacity):
            if demands[node] > remaining_capacity:
                continue

            # Try inserting the node at all possible positions in the vehicle's route
            route = current_solution.routes[vehicle_id]
            for position in range(len(route) + 1):
                prev_node = depot if position == 0 else route[position - 1]
                next_node = depot if position == len(route) else route[position]

                # Calculate the cost increase for this insertion
                cost_increase = (
                    distance_matrix[prev_node][node] +
                    distance_matrix[node][next_node] -
                    distance_matrix[prev_node][next_node]
                )

                # Calculate the penalty for spread (distance from depot to farthest node in the route)
                farthest_distance = max(
                    [distance_matrix[depot][n] for n in route + [node] if route] + [0]
                )
                spread_penalty = spread_penalty_weight * farthest_distance

                # Factor in vehicle capacity utilization
                capacity_utilization_penalty = capacity_utilization_weight * (1 - remaining_capacity / capacity)

                # Combine the metrics into a single score
                score = cost_increase + spread_penalty + capacity_utilization_penalty

                if score < best_score:
                    best_score = score
                    best_node = node
                    best_vehicle = vehicle_id
                    best_position = position

    if best_node is not None and best_vehicle is not None and best_position is not None:
        # Perform the insertion
        operator = InsertOperator(best_vehicle, best_node, best_position)

        # Step 2: Periodic greedy improvement
        if len(unvisited_nodes) % apply_greedy_frequency == 0:
            best_delta = 0
            best_vehicle = None
            best_segment = None

            for vehicle_id, route in enumerate(current_solution.routes):
                if len(route) < 2:
                    continue

                # Evaluate all 2-opt swaps for compactness
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        if j == len(route) - 1 and i == 0:
                            continue

                        before = distance_matrix[depot if i == 0 else route[i - 1]][route[i]]
                        after = distance_matrix[depot if j == len(route) - 1 else route[j]][route[j - 1]]
                        new_before = distance_matrix[depot if i == 0 else route[i - 1]][route[j - 1]]
                        new_after = distance_matrix[depot if j == len(route) - 1 else route[j]][route[i]]
                        delta = (new_before + new_after) - (before + after)

                        if delta < best_delta:
                            best_delta = delta
                            best_vehicle = vehicle_id
                            best_segment = (i, j)

            if best_vehicle is not None and best_segment is not None:
                i, j = best_segment
                return ReverseSegmentOperator(best_vehicle, [(i, j - 1)]), {}

        # Step 3: Periodic inter-vehicle swapping
        if len(unvisited_nodes) % apply_swap_frequency == 0:
            best_cost_reduction = float('-inf')
            best_source_vehicle_id = None
            best_source_position = None
            best_target_vehicle_id = None
            best_target_position = None

            for source_vehicle_id, source_route in enumerate(current_solution.routes):
                for source_position, node in enumerate(source_route):
                    # Skip if no nodes to shift
                    if not source_route:
                        continue

                    # Calculate the load after removing the node
                    new_load_source = vehicle_remaining_capacity[source_vehicle_id] + demands[node]
                    if new_load_source > capacity:
                        continue  # Skip if moving the node violates source vehicle's capacity

                    # Check each target route to find the best shift
                    for target_vehicle_id, target_route in enumerate(current_solution.routes):
                        if source_vehicle_id == target_vehicle_id:
                            continue

                        new_load_target = vehicle_remaining_capacity[target_vehicle_id] - demands[node]
                        if new_load_target < 0:
                            continue  # Skip if moving the node violates target vehicle's capacity

                        for target_position in range(len(target_route) + 1):
                                # Calculate the cost difference if the node is inserted at the target position
                                source_previous_node = depot if source_position == 0 else source_route[source_position - 1]
                                source_next_node = depot if source_position + 1 == len(source_route) else source_route[source_position + 1]
                                target_previous_node = depot if target_position == 0 else target_route[target_position - 1]
                                target_next_node = depot if target_position == len(target_route) else target_route[target_position]

                                cost_increase = (
                                    -distance_matrix[source_previous_node][node]
                                    -distance_matrix[node][source_next_node]
                                    +distance_matrix[source_previous_node][source_next_node]
                                    +distance_matrix[target_previous_node][node]
                                    +distance_matrix[node][target_next_node]
                                    -distance_matrix[target_previous_node][target_next_node]
                                )
                                cost_reduction = -cost_increase

                                # Update best shift if this shift is better
                                if cost_reduction > best_cost_reduction:
                                    best_source_vehicle_id = source_vehicle_id
                                    best_source_position = source_position
                                    best_target_vehicle_id = target_vehicle_id
                                    best_target_position = target_position
                                    best_cost_reduction = cost_reduction

            # If a beneficial swap is found, return the corresponding operator
            if best_cost_reduction > 0:
                return RelocateOperator(
                    source_vehicle_id=best_source_vehicle_id,
                    source_position=best_source_position,
                    target_vehicle_id=best_target_vehicle_id,
                    target_position=best_target_position
                ), {}

        # Return the best insertion operator
        return operator, {}

    # If no valid insertion was found
    return None, {}