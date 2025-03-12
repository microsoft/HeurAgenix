from src.problems.dposp.components import *
import numpy as np

def shortest_operation_ae31(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """Shortest Operation Heuristic with Optimization (AE31) for DPOSP.

    This heuristic aims to maximize the number of fulfilled orders by prioritizing orders based on deadlines, feasibility, and production times. 
    It dynamically searches for better insertion positions, shifts orders, and periodically optimizes through swaps.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "production_line_num" (int): Total number of production lines.
            - "order_num" (int): Total number of orders.
            - "order_quantity" (list[int]): Quantity required for each order.
            - "order_deadline" (list[float]): Deadline for each order.
            - "order_product" (list[int]): Product associated with each order.
            - "production_rate" (numpy.array): 2D array of production speeds for each product on each production line.
            - "transition_time" (numpy.array): 3D array of transition times between products for each production line.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): Current scheduling solution.
            - "unfulfilled_orders" (list[int]): List of unfulfilled orders.
            - "feasible_orders_to_fulfill" (list[int]): List of feasible orders that can be fulfilled.
            - "validation_single_production_schedule" (callable): Function to validate a single production schedule.
            - "get_time_cost_delta" (callable): Function to compute the time cost delta for inserting an order.
            - "total_time_cost_per_production_line" (list[float]): Total time cost for each production line.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This heuristic does not use algorithm_data.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.
        **kwargs: Hyper-parameters for the heuristic:
            - swap_frequency (int, default=10): Frequency (in terms of operations) at which swap optimization is performed.
            - shift_frequency (int, default=5): Frequency (in terms of operations) at which order shifting is performed.

    Returns:
        InsertOperator: The operator to insert the selected order into the selected production line.
        dict: Updated algorithm data, if any.
    """
    # Extract necessary data from global_data and state_data
    production_line_num = global_data["production_line_num"]
    order_deadline = global_data["order_deadline"]
    order_quantity = global_data["order_quantity"]
    order_product = global_data["order_product"]
    production_rate = global_data["production_rate"]
    transition_time = global_data["transition_time"]

    current_solution = state_data["current_solution"]
    feasible_orders_to_fulfill = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]
    get_time_cost_delta = state_data["get_time_cost_delta"]

    # Hyper-parameters
    swap_frequency = kwargs.get("swap_frequency", 10)
    shift_frequency = kwargs.get("shift_frequency", 5)
    operation_count = algorithm_data.get("operation_count", 0)

    # Return None if no feasible orders are available
    if not feasible_orders_to_fulfill:
        return None, {"operation_count": operation_count}

    # Step 1: Prioritize feasible orders based on deadlines and quantities
    sorted_orders = sorted(
        feasible_orders_to_fulfill,
        key=lambda order: (order_deadline[order], order_quantity[order])
    )

    # Step 2: Search for the best insertion position for each feasible order
    best_operator = None
    min_time_cost = float('inf')

    for order_id in sorted_orders:
        product_id = order_product[order_id]
        for line_id in range(production_line_num):
            # Skip if the production line cannot produce this product
            if production_rate[line_id][product_id] == 0:
                continue

            current_schedule = current_solution.production_schedule[line_id]

            for position in range(len(current_schedule) + 1):
                # Validate the potential new schedule
                new_schedule = current_schedule[:]
                new_schedule.insert(position, order_id)

                if not validation_single_production_schedule(line_id, new_schedule):
                    continue

                # Calculate the time cost delta
                delta_time_cost = get_time_cost_delta(line_id, order_id, position)
                if delta_time_cost < min_time_cost:
                    min_time_cost = delta_time_cost
                    best_operator = InsertOperator(production_line_id=line_id, order_id=order_id, position=position)

    # Step 3: Perform order shifting periodically
    if operation_count % shift_frequency == 0:
        for source_line_id, source_schedule in enumerate(current_solution.production_schedule):
            for order_id in source_schedule:
                for target_line_id in range(production_line_num):
                    if source_line_id == target_line_id or production_rate[target_line_id][order_product[order_id]] == 0:
                        continue

                    for position in range(len(current_solution.production_schedule[target_line_id]) + 1):
                        # Attempt to shift the order and validate schedules
                        trial_target_schedule = current_solution.production_schedule[target_line_id][:]
                        trial_source_schedule = source_schedule[:]
                        trial_target_schedule.insert(position, order_id)
                        trial_source_schedule.remove(order_id)

                        if not validation_single_production_schedule(source_line_id, trial_source_schedule) or not validation_single_production_schedule(target_line_id, trial_target_schedule):
                            continue

                        # Calculate time cost
                        delta_time_cost = get_time_cost_delta(target_line_id, order_id, position)
                        if delta_time_cost < min_time_cost and current_solution.routes[source_line_id][source_schedule.index(order_id)] != current_solution.depot:
                            min_time_cost = delta_time_cost
                            best_operator = RelocateOperator(
                                source_production_line_id=source_line_id,
                                source_position=source_schedule.index(order_id),
                                target_production_line_id=target_line_id,
                                target_position=position
                            )

    # Step 4: Perform swap optimization periodically
    if operation_count % swap_frequency == 0:
        for line_id, schedule in enumerate(current_solution.production_schedule):
            for i in range(len(schedule) - 1):
                for j in range(i + 1, len(schedule)):
                    # Attempt to swap and validate the new schedule
                    new_schedule = schedule[:]
                    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                    if validation_single_production_schedule(line_id, new_schedule):
                        delta_time_cost = get_time_cost_delta(line_id, schedule[j], i) + get_time_cost_delta(line_id, schedule[i], j)
                        if delta_time_cost < min_time_cost:
                            min_time_cost = delta_time_cost
                            best_operator = SwapOperator(
                                production_line_id1=line_id, position1=i,
                                production_line_id2=line_id, position2=j
                            )

    # Update operation count and return the best operator found
    if best_operator:
        return best_operator, {"operation_count": operation_count + 1}
    return None, {"operation_count": operation_count + 1}