from src.problems.base.mdp_components import Solution, ActionOperator

def demand_responsive_dispatch_fc99(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm that allocates available EVs to trip requests and charging based on real-time customer arrival data and SoC levels.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV.
            - battery_soc (list[float]): A list representing the battery state of charge for each EV.
            - time_to_next_availability (list[int]): A list indicating the time until each EV becomes available.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - (No specific items needed for this function currently)
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.

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

    # Prioritize EVs that are currently serving rides and have low SoC, and are about to become available
    prioritize_for_charging = [i for i in range(fleet_size) if operational_status[i] == 1 and time_to_next_availability[i] == 0 and battery_soc[i] < 0.3]

    # List of idle EVs eligible for charging
    eligible_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] == 0 and i not in prioritize_for_charging]

    # Combine prioritized EVs with eligible EVs
    prioritized_evs = prioritize_for_charging + eligible_evs

    # Sort prioritized EVs based on SoC in ascending order (prioritize low SoC for charging)
    prioritized_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in range(min(len(prioritized_evs), total_chargers)):
        actions[prioritized_evs[i]] = 1

    # Ensure no EV serving a ride is assigned a charging action
    actions = [0 if time_to_next_availability[i] > 0 else actions[i] for i in range(fleet_size)]

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, {}