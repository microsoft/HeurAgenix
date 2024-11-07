from src.problems.dposp.components import *

def order_shift_between_lines_bd0c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[RelocateOperator, dict]:
    """
    This heuristic attempts to shift an unfulfilled order from one production line to another while adhering to machine capabilities, transition rules, and order deadlines.
    
    Args:
        global_data (dict): Contains the global static information about the problem.
        state_data (dict): Contains the current dynamic state data of the problem.
        algorithm_data (dict): Contains data specific to the algorithm (unused in this heuristic).
        get_state_data_function (callable): Function that generates state data for a given solution.
    
    Returns:
        (RelocateOperator, dict): The operator to shift an order between production lines and an empty dictionary, as the heuristic does not update algorithm_data.
    """
    # Retrieve necessary data from global_data
    production_rate = global_data["production_rate"]
    transition_time = global_data["transition_time"]
    
    # Retrieve necessary data from state_data
    current_solution = state_data["current_solution"]
    feasible_orders_to_fulfill = state_data["feasible_orders_to_fulfill"]
    validation_single_production_schedule = state_data["validation_single_production_schedule"]
    
    # Set a default value for optional hyper parameters
    max_attempts = kwargs.get('max_attempts', 100)

    # Initialize variables to store the best shift found
    best_order_id = None
    best_source_line_id = None
    best_target_line_id = None
    best_position = None
    best_delta_time_cost = float('inf')

    # Iterate through each feasible order to find a beneficial shift
    for source_line_id, source_schedule in enumerate(current_solution.production_schedule):
        for order_id in source_schedule:
            for target_line_id, target_schedule in enumerate(current_solution.production_schedule):
                # Skip if it's the same production line or the line cannot produce the product
                if source_line_id == target_line_id or production_rate[target_line_id][global_data["order_product"][order_id]] == 0:
                    continue

                # Find the best position to insert the order in the target production line
                for position in range(len(target_schedule) + 1):
                    # Copy the current schedule and try inserting the order
                    trial_target_schedule = target_schedule[:]
                    trial_target_schedule.insert(position, order_id)
                    trial_source_schedule = source_schedule[:]
                    trial_source_schedule.remove(order_id)

                    # Validate the trial schedule
                    if not validation_single_production_schedule(source_line_id, trial_source_schedule) or not validation_single_production_schedule(target_line_id, trial_target_schedule):
                        continue  # Skip if the new schedule is not valid

                    # Calculate the delta time cost of the new schedule
                    new_production_schedule = [schedule[:] for schedule in current_solution.production_schedule]
                    new_production_schedule[target_line_id] = trial_target_schedule
                    new_production_schedule[source_line_id] = trial_source_schedule
                    state_data_for_trial = get_state_data_function(Solution(new_production_schedule))
                    delta_time_cost = state_data_for_trial["total_time_cost_per_production_line"][target_line_id] + state_data_for_trial["total_time_cost_per_production_line"][source_line_id] - state_data["total_time_cost_per_production_line"][target_line_id] - state_data["total_time_cost_per_production_line"][source_line_id]

                    # Check if this shift leads to a better solution
                    if delta_time_cost < best_delta_time_cost:
                        best_order_id = order_id
                        best_source_line_id = source_line_id
                        best_target_line_id = target_line_id
                        best_position = position
                        best_delta_time_cost = delta_time_cost

    # If a beneficial shift is found, create and return the corresponding operator
    if best_order_id is not None:
        return RelocateOperator(
            source_production_line_id=best_source_line_id,
            source_position=state_data["current_solution"].production_schedule[best_source_line_id].index(best_order_id),
            target_production_line_id=best_target_line_id,
            target_position=best_position
        ), {}

    # If no beneficial shift is found, return None
    return None, {}