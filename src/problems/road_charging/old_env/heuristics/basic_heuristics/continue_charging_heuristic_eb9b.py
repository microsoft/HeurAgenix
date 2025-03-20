from src.problems.base.mdp_components import Solution, ActionOperator

def continue_charging_heuristic_eb9b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic prioritizes vehicles already charging to continue charging to avoid initial connection costs.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "charging_lead_time" (list[int]): Charging leading time for each vehicle.
            - "battery_soc" (list[float]): State of charge for each vehicle.
        get_state_data_function (callable): The function that receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.

    Returns:
        An ActionOperator with prioritized charging actions for vehicles already charging.
        An empty dictionary as this heuristic does not update the algorithm data.
    """
    fleet_size = len(state_data["ride_lead_time"])
    actions = [0] * fleet_size  # Initialize actions to 0 (no charging)

    # Loop through each vehicle to determine actions
    for i in range(fleet_size):
        # Check if the vehicle is on a ride and can't charge
        if state_data["ride_lead_time"][i] >= 2:
            actions[i] = 0
            continue

        # Check if the vehicle is already charging and battery is not full
        if state_data["charging_lead_time"][i] > 0 and state_data["battery_soc"][i] < 1:
            actions[i] = 1

        # Ensure vehicles that are fully charged do not charge
        if state_data["battery_soc"][i] >= 1:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > global_data["total_chargers"]:
        # Prioritize vehicles with the longest charging lead time
        charging_priority = sorted(range(fleet_size), key=lambda x: state_data["charging_lead_time"][x], reverse=True)
        for idx in charging_priority:
            if sum(actions) <= global_data["total_chargers"]:
                break
            actions[idx] = 0

    # Return the ActionOperator with the decided actions
    return ActionOperator(actions), {}