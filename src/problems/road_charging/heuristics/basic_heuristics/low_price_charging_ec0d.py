from src.problems.base.mdp_components import Solution, ActionOperator

def low_price_charging_ec0d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to prioritize charging during low-cost periods.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
            - "charging_price" (list[float]): Charging prices at each time step.
            - "max_time_steps" (int): Maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): State of charge for each EV in the fleet.
            - "ride_lead_time" (list[int]): Remaining ride time for each EV.
            - "charging_lead_time" (list[int]): Current charging time for each EV.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the origin solution.
        (Optional) **kwargs: Hyper-parameters for the algorithm. Default values are set if not provided.
            - "low_battery_threshold" (float): The threshold below which an EV is considered low on battery. Default is 0.1 (10%).

    Returns:
        ActionOperator: An operator to modify the solution by prioritizing charging during low-cost periods.
        dict: Empty dictionary, as no additional algorithm data needs to be updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]

    # Hyper-parameters
    low_battery_threshold = kwargs.get("low_battery_threshold", 0.1)

    # Initialize actions for this time step
    actions = [0] * len(battery_soc)

    # Determine actions based on battery SoC and ride lead time
    chargers_used = sum(1 for clt in charging_lead_time if clt > 0)
    for ev_index, soc in enumerate(battery_soc):
        if chargers_used >= total_chargers:
            break
        if soc < low_battery_threshold and ride_lead_time[ev_index] == 0 and charging_lead_time[ev_index] == 0:
            actions[ev_index] = 1
            chargers_used += 1

    # Return the action operator with the determined actions
    return ActionOperator(actions), {}