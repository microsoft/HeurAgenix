from src.problems.base.mdp_components import Solution, ActionOperator

def maintain_minimum_so_c_16c0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to ensure EVs charge to maintain a minimum SoC before ride assignments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): SoC of the battery of each EV.
            - "ride_lead_time" (list[int]): Remaining ride time for each EV, 0 if idle.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the original solution.
        kwargs: No additional hyperparameters are required for this heuristic.

    Returns:
        ActionOperator: Operator to apply actions ensuring minimum SoC.
        dict: Empty dictionary as there is no update to algorithm data in this heuristic.
    """
    fleet_size = global_data["fleet_size"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Initialize actions with all 0s (default to not charging).
    actions = [0] * fleet_size

    # Determine which EVs should charge based on their SoC.
    for i in range(fleet_size):
        if ride_lead_time[i] == 0:  # Only consider EVs that are not currently on a ride.
            if battery_soc[i] < 0.1:
                actions[i] = 1  # Prioritize charging if SoC is below threshold.

    # Create and return the ActionOperator with the determined actions.
    return ActionOperator(actions), {}