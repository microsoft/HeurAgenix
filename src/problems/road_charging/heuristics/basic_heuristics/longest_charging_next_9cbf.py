from src.problems.base.mdp_components import ActionOperator

def longest_charging_next_9cbf(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic prioritizes the EV with the longest remaining charging requirement to ensure that vehicles requiring more energy are given priority access to charging resources.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The total number of EVs in the fleet.
            - "total_chargers" (int): The total number of chargers available.
            - "max_cap" (float): The maximum battery capacity for any EV.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): State of charge (SoC) for each EV in the fleet.
            - "ride_lead_time" (list[int]): Remaining time steps for each EV that is currently on a ride.
        
    Returns:
        ActionOperator: An instance of ActionOperator that schedules the next charging action for the EV with the maximum remaining charging requirement.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Calculate remaining charging requirement for each EV
    remaining_charging_requirement = [global_data["max_cap"] - soc for soc in battery_soc]

    # Initialize actions list with zeros
    actions = [0] * fleet_size

    # Sort EVs based on remaining charging requirement, descending
    ev_indices = sorted(range(fleet_size), key=lambda i: remaining_charging_requirement[i], reverse=True)

    chargers_used = 0
    for i in ev_indices:
        if chargers_used >= total_chargers:
            break
        if battery_soc[i] >= 1 or ride_lead_time[i] >= 2:
            continue

        # Schedule this EV for charging
        actions[i] = 1
        chargers_used += 1

    return ActionOperator(actions), {}