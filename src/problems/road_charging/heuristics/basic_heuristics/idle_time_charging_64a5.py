from src.problems.base.mdp_components import Solution, ActionOperator

def idle_time_charging_64a5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ The IdleTimeCharging heuristic focuses on scheduling charging during predicted idle times for EVs.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
            - "assign_prob" (list[float]): Probability of receiving a ride order when the vehicle is in idle status for each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle, with length equal to fleet_size.
            - "battery_soc" (list[float]): State of charge of the battery for each vehicle, with length equal to fleet_size.

    Returns:
        An ActionOperator that contains the charging actions for each EV.
        An empty dictionary as no algorithm data is updated in this heuristic.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    assign_prob = global_data["assign_prob"]
    current_step = state_data["current_step"]

    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions for each vehicle to 0 (not charging)
    actions = [0] * fleet_size

    # Sort vehicles by state of charge to prioritize lower SoC for charging
    vehicles_sorted_by_soc = sorted(range(fleet_size), key=lambda i: battery_soc[i])

    chargers_available = total_chargers

    for i in vehicles_sorted_by_soc:
        # Skip if the vehicle is on a ride (ride_lead_time >= 2)
        if ride_lead_time[i] >= 2:
            continue

        # Skip if the vehicle's battery is fully charged
        if battery_soc[i] >= 1:
            continue

        # Check if chargers are available
        if chargers_available <= 0:
            break

        # Assign charging if the vehicle is idle, prioritizing low SoC vehicles
        if ride_lead_time[i] == 0:
            actions[i] = 1
            chargers_available -= 1

    # Ensure the sum of actions does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset actions if over the limit

    return ActionOperator(actions), {}