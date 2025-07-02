from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_2cfe(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm prioritizes charging EVs with the lowest battery SoC.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - max_time_steps (int): The maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
        threshold (float, optional): The dynamic battery SoC threshold calculated based on current_step. Default is 0.7.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    max_time_steps = global_data["max_time_steps"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Calculate dynamic threshold
    threshold = kwargs.get("threshold", max(0.7, 0.5 + 0.2 * (current_step / max_time_steps)))

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions based on battery SoC and operational status
    for i in range(fleet_size):
        # If the EV is idle and its SoC is below the dynamic threshold, consider charging
        if operational_status[i] == 0 and battery_soc[i] < threshold:
            actions[i] = 1

    # Sort EV indices by battery SoC to prioritize those with the lowest SoC
    soc_indices = np.argsort(battery_soc)

    # Ensure the number of charging actions does not exceed the total chargers
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