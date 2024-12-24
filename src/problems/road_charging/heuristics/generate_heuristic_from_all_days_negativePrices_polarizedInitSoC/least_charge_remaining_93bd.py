from src.problems.base.mdp_components import ActionOperator
import numpy as np

def least_charge_remaining_93bd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic prioritizes scheduling charging actions for EVs with the lowest remaining battery state of charge (SoC).

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
            - "fleet_size" (int): Number of EVs in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): Current state of charge (SoC) for each EV in the fleet.
            - "ride_lead_time" (list[int]): Remaining ride time for each EV. An EV cannot charge if it is currently on a ride.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator: Operator to schedule charging actions for selected EVs.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Initialize action list with all zeros, meaning no EV is scheduled for charging by default.
    actions = [0] * global_data["fleet_size"]

    # Extract necessary data from state_data
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Identify EVs that can be scheduled for charging (not on ride and not fully charged)
    ev_indices = [
        i for i in range(global_data["fleet_size"])
        if ride_lead_time[i] < 2 and battery_soc[i] < 1
    ]

    # Sort EVs by SoC in ascending order (prioritize those with the lowest SoC)
    ev_indices.sort(key=lambda i: battery_soc[i])

    # Schedule EVs for charging until all chargers are occupied or no more EVs need charging
    num_chargers = global_data["total_chargers"]
    for i in ev_indices:
        if sum(actions) < num_chargers:
            actions[i] = 1

    # Return the operator to apply these actions
    return ActionOperator(actions), {}