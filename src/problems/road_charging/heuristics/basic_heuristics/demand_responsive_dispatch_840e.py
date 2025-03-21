from src.problems.base.mdp_components import Solution, ActionOperator

def demand_responsive_dispatch_840e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Heuristic algorithm that allocates available EVs to trip requests based on real-time customer arrival data.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "total_chargers" (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV becomes available.
        algorithm_data (dict): Not necessary for this algorithm.
        get_state_data_function (callable): Function that receives the new solution and returns the state dictionary for the new solution.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step.
        dict: Updated algorithm data. Empty in this case.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]

    # Initialize action list for all EVs
    actions = [0] * fleet_size

    # List of EVs eligible for charging (idle and not serving a trip)
    eligible_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] == 0]

    # Sort eligible EVs based on SoC in ascending order (prioritize low SoC for charging)
    eligible_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in range(min(len(eligible_evs), total_chargers)):
        actions[eligible_evs[i]] = 1

    # Ensure no EV serving a ride is assigned a charging action
    actions = [0 if time_to_next_availability[i] > 0 else actions[i] for i in range(fleet_size)]

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, {}