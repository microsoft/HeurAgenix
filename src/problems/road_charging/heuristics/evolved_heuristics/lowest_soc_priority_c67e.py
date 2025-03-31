from src.problems.base.mdp_components import ActionOperator
import numpy as np

def lowest_soc_priority_c67e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV charging prioritization based on fleet size and charger availability.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory (solution) for EVs.
            - battery_soc (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - operational_status (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.
        
        kwargs:
            - fleet_threshold (int): The threshold fleet size to switch prioritization strategy. Default is 5.
            - average_customer_arrivals (float): The average number of customer arrivals, used for priority calculation.

    Returns:
        ActionOperator: The operator assigning charging actions to EVs.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    operational_status = state_data["operational_status"]
    fleet_threshold = kwargs.get("fleet_threshold", 5)

    # Initialize actions with zeros
    actions = [0] * fleet_size

    if fleet_size <= fleet_threshold:
        # Original logic: Prioritize EVs with the lowest SoC
        chargeable_evs = [
            i for i in range(fleet_size)
            if operational_status[i] == 0
        ]
        chargeable_evs.sort(key=lambda i: battery_soc[i])
    else:
        # New logic: Prioritize EVs with higher SoC when fleet size is large
        priority_scores = [
            (battery_soc[i] / (fleet_size / total_chargers), i)
            for i in range(fleet_size) if operational_status[i] == 0
        ]
        chargeable_evs = [i for _, i in sorted(priority_scores, reverse=True)]

    # Assign charging actions based on priority
    for i in chargeable_evs[:total_chargers]:
        actions[i] = 1

    # Ensure the sum of actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    operator = ActionOperator(actions)
    return operator, {}