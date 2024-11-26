from src.problems.base.mdp_components import Solution, ActionOperator

def earliest_deadline_first_3204(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Prioritize charging EVs that are closest to their next ride deadline to ensure they are ready when needed.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): The remaining time until the next ride for each EV.
            - "battery_soc" (list[float]): The current state of charge for each EV.
            - "reward" (float): The total reward for this time step.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - No specific algorithm data is required.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        An ActionOperator that updates the solution with charging decisions based on the earliest deadline first heuristic.
        An empty dictionary as no algorithm data updates are necessary.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions with all 0s (idle)
    actions = [0] * len(ride_lead_time)

    # Collect indices of EVs that are not currently on a ride
    idle_indices = [i for i, lead_time in enumerate(ride_lead_time) if lead_time == 0]

    # Sort idle EVs by their ride deadline (earliest first)
    prioritized_indices = sorted(idle_indices, key=lambda i: ride_lead_time[i])

    # Determine how many EVs can be charged (based on available chargers)
    num_to_charge = min(total_chargers, len(prioritized_indices))

    # Assign charging action (1) to the selected EVs
    for i in range(num_to_charge):
        actions[prioritized_indices[i]] = 1

    # Return the ActionOperator with the updated actions
    return ActionOperator(actions), {}