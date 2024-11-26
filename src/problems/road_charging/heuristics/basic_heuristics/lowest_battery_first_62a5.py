from src.problems.base.mdp_components import Solution, ActionOperator

def lowest_battery_first_62a5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Prioritize charging EVs with the lowest battery state of charge (SoC) to prevent any from running out of power.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): List of current battery state of charge (SoC) for each vehicle.
            - "ride_lead_time" (list[int]): List indicating the remaining ride time for each vehicle.
        algorithm_data (dict): Not utilized in this algorithm.
        get_state_data_function (callable): Not utilized in this algorithm.
        **kwargs: No additional hyper parameters are used in this algorithm.

    Returns:
        An ActionOperator with actions prioritizing charging for EVs with the lowest SoC.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    # Extract necessary data from the input dictionaries
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Initialize action list with all 0 (no charging) for each EV
    actions = [0] * fleet_size

    # Create a list of indices for EVs that are not currently on a ride
    eligible_indices = [i for i in range(fleet_size) if ride_lead_time[i] == 0]

    # Sort eligible EV indices by their battery state of charge (SoC) in ascending order
    sorted_indices = sorted(eligible_indices, key=lambda i: battery_soc[i])

    # Prioritize charging for the top M EVs with the lowest SoC
    for i in range(min(total_chargers, len(sorted_indices))):
        actions[sorted_indices[i]] = 1

    # Create and return the action operator for the modified actions
    operator = ActionOperator(actions)
    return operator, {}