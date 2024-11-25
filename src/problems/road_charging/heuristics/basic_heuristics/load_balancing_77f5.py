from src.problems.base.mdp_components import Solution, ActionOperator

def load_balancing_77f5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic algorithm aims to distribute charging requests evenly across available chargers to prevent any single charger from being overwhelmed, maintaining overall system efficiency.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The number of total chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Current ride lead time for each EV, indicates if EV is on a ride.
            - "battery_soc" (list[int]): Current state of charge for each EV in the fleet.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.
            - "min_soc_threshold" (int): The minimum state of charge threshold to consider charging an EV. Default is 20.

    Returns:
        An ActionOperator with actions that distribute charging evenly.
        An empty dictionary for algorithm data as no additional data needs to be updated.
    """
    # Retrieve necessary data
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]
    min_soc_threshold = kwargs.get("min_soc_threshold", 20)

    # Initialize actions with all zeros (no charging)
    actions = [0] * len(battery_soc)

    # Count currently charging EVs
    currently_charging = sum([1 for soc in battery_soc if soc < min_soc_threshold])

    # Calculate available chargers
    available_chargers = total_chargers - currently_charging

    # If no chargers are available, return an empty operator
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    # Distribute charging requests to EVs with lowest SoC, not on a ride
    charge_candidates = [(i, soc) for i, soc in enumerate(battery_soc) if ride_lead_time[i] == 0 and soc < min_soc_threshold]
    charge_candidates.sort(key=lambda x: x[1])  # Sort by SoC

    # Assign charging actions to the EVs with the lowest SoC
    for i, (ev_index, soc) in enumerate(charge_candidates):
        if i < available_chargers:
            actions[ev_index] = 1
        else:
            break

    # Return the ActionOperator with the updated actions
    return ActionOperator(actions), {}