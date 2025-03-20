from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_50ec(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm prioritizes charging EVs with low battery SoC.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
        algorithm_data (dict): Not used in this heuristic.
        get_state_data_function (callable): Not used in this heuristic.
        threshold (float, optional): The battery SoC threshold below which an EV should prioritize charging. Default is 0.7.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    threshold = kwargs.get("threshold", 0.7)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions based on battery SoC and operational status
    for i in range(fleet_size):
        # If the EV is on a ride, it cannot charge
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # If the EV is idle and its SoC is below the threshold, consider charging
        elif operational_status[i] == 0 and battery_soc[i] < threshold:
            actions[i] = 1

    # Ensure the number of charging actions does not exceed the total chargers
    if sum(actions) > total_chargers:
        # Select EVs with the lowest SoC to charge
        soc_indices = np.argsort(battery_soc)
        chargers_used = 0
        for idx in soc_indices:
            if chargers_used < total_chargers and operational_status[idx] == 0 and battery_soc[idx] < threshold:
                actions[idx] = 1
                chargers_used += 1
            else:
                actions[idx] = 0

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, {}