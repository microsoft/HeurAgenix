from src.problems.base.mdp_components import Solution, ActionOperator

def so_c_threshold_1467(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Set a minimum SoC threshold for charging initiation to avoid frequent short charging sessions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): The state of charge (SoC) of each vehicle's battery.
            - "ride_lead_time" (list[int]): The remaining time for vehicles currently on a ride.
            - "current_step" (int): The current time step in the simulation.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        introduction for hyper parameters in kwargs if used:
            - "soc_threshold" (float): The minimum SoC threshold for initiating charging. Default is 0.2 (20%).

    Returns:
        An ActionOperator that applies the calculated actions.
        An empty dictionary as there is no update to the algorithm data.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    
    # Set default hyper-parameter value
    soc_threshold = kwargs.get("soc_threshold", 0.2)

    # Initialize actions for all vehicles as 0 (not charging)
    actions = [0] * len(battery_soc)

    # Create a list of tuples (index, soc) for vehicles eligible for charging
    eligible_vehicles = [
        (i, soc) for i, soc in enumerate(battery_soc)
        if soc < soc_threshold and ride_lead_time[i] < 2
    ]

    # Sort eligible vehicles by SoC (ascending order)
    eligible_vehicles.sort(key=lambda x: x[1])

    # Assign charging actions to the lowest SoC vehicles within the charger limit
    chargers_used = 0
    for i, soc in eligible_vehicles:
        if chargers_used < total_chargers:
            actions[i] = 1
            chargers_used += 1
        else:
            break

    # Return the action operator and an empty dictionary for algorithm data
    return ActionOperator(actions), {}