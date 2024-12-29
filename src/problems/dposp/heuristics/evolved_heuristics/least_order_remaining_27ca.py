from src.problems.dposp.components import *
import numpy as np

def least_order_remaining_27ca(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """ Heuristic to prioritize orders based on a weighted score incorporating deadline proximity, production feasibility, and potential to maximize fulfilled orders.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production speeds for each product on each production line.
            - order_deadline (numpy.array): 1D array of deadlines for each order.
            - order_quantity (numpy.array): 1D array of quantities required for each order.
            - order_product (numpy.array): 1D array mapping each order to its product.
            - production_line_num (int): Total number of production lines.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current scheduling solution.
            - feasible_orders_to_fulfill (list[int]): List of feasible orders that can be fulfilled.
            - validation_single_production_schedule (callable): Function to validate a single production schedule.
            - get_time_cost_delta (callable): Function to compute the time cost delta for inserting an order.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - operation_count (int): The number of operations performed so far. Defaults to 0.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the origin solution.
        kwargs: Hyper-parameters for the heuristic:
            - swap_frequency (int, default=10): Frequency (in terms of operations) at which swap optimization is performed.
            - shift_frequency (int, default=5): Frequency (in terms of operations) at which order shifting is performed.

    Returns:
        AppendOperator: The operator to append the selected order to a production line's schedule.
        dict: Updated algorithm data, if any.
    """
    # Extract necessary data from global_data and state_data
    production_rate = global_data["production_rate"]
    order_deadline = global_data["order_deadline"]
    order_quantity = global_data["order_quantity"]
    order_product = global_data["order_product"]
    production_line_num = global_data["production_line_num"]

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

    # Step 1: Calculate scores for feasible orders
    best_operator = None
    max_score = -float('inf')

    for order_id in feasible_orders_to_fulfill:
        product_id = order_product[order_id]
        deadline_weight = 1 / (order_deadline[order_id] + 1)
        quantity_penalty = 1 / order_quantity[order_id]

        for line_id in range(production_line_num):
            # Skip if the production line cannot produce this product
            if production_rate[line_id][product_id] == 0:
                continue

            current_schedule = current_solution.production_schedule[line_id]

            # Simulate appending the order at the end of the production line
            new_schedule = current_schedule + [order_id]

            # Validate the new schedule
            if not validation_single_production_schedule(line_id, new_schedule):
                continue

            # Calculate feasibility factor for this order on this production line
            feasibility_factor = 1 / (len(current_schedule) + 1)  # Simple normalization

            # Calculate the score for this order
            score = deadline_weight * quantity_penalty * feasibility_factor

            # Update the best operator if the score is higher
            if score > max_score:
                max_score = score
                best_operator = AppendOperator(production_line_id=line_id, order_id=order_id)

    # Step 2: Perform order shifting periodically
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
                        if delta_time_cost < 0:  # Improvement
                            best_operator = RelocateOperator(
                                source_production_line_id=source_line_id,
                                source_position=source_schedule.index(order_id),
                                target_production_line_id=target_line_id,
                                target_position=position
                            )

    # Step 3: Perform swap optimization periodically
    if operation_count % swap_frequency == 0:
        for line_id, schedule in enumerate(current_solution.production_schedule):
            for i in range(len(schedule) - 1):
                for j in range(i + 1, len(schedule)):
                    # Attempt to swap and validate the new schedule
                    new_schedule = schedule[:]
                    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                    if validation_single_production_schedule(line_id, new_schedule):
                        delta_time_cost = get_time_cost_delta(line_id, schedule[j], i) + get_time_cost_delta(line_id, schedule[i], j)
                        if delta_time_cost < 0:  # Improvement
                            best_operator = SwapOperator(
                                production_line_id1=line_id, position1=i,
                                production_line_id2=line_id, position2=j
                            )

    # Update operation count and return the best operator found
    if best_operator:
        return best_operator, {"operation_count": operation_count + 1}
    return None, {"operation_count": operation_count + 1}