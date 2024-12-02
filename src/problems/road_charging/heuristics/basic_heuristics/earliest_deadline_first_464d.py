from src.problems.base.mdp_components import Solution, ActionOperator

def earliest_deadline_first_464d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Schedule EVs with the lowest remaining battery SoC to charge first.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time for each EV, indicating if they are on a ride.
            - "battery_soc" (list[float]): State of charge (SoC) of each EV.
        kwargs: Hyper-parameters for the algorithm:
            - "low_battery_threshold" (float, default=0.1): Threshold for low battery SoC.

    Returns:
        ActionOperator: An operator that modifies the solution to charge EVs with the lowest SoC.
        dict: An empty dictionary, as no algorithm-specific data is updated.
    """
    # Extract necessary data from global_data and state_data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Extract hyper-parameters from kwargs
    low_battery_threshold = kwargs.get("low_battery_threshold", 0.1)

    # Initialize actions with all 0s (no charging)
    actions = [0] * fleet_size

    # Create a list of EVs that are not on a ride and have low battery
    eligible_evs = [(i, soc) for i, (ride, soc) in enumerate(zip(ride_lead_time, battery_soc)) if ride == 0]

    # Sort eligible EVs by their SoC in ascending order
    eligible_evs.sort(key=lambda x: x[1])

    # Select EVs to charge, prioritizing those with SoC below the threshold
    chargers_available = total_chargers
    for ev_id, soc in eligible_evs:
        if chargers_available <= 0:
            break
        if soc < low_battery_threshold or len(eligible_evs) <= total_chargers:
            actions[ev_id] = 1
            chargers_available -= 1

    # Create and return the ActionOperator with the new actions
    operator = ActionOperator(actions)
    return operator, {}