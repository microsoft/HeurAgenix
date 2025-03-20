from src.problems.base.mdp_components import Solution, ActionOperator

def no_charging_heuristic_446c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Determine if no vehicles should be charging based on factors such as high charging costs or low remaining time in the scheduling horizon.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle in the fleet.
            - "battery_soc" (list[float]): State of charge for each vehicle in the fleet.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        **kwargs: Any additional hyper parameters, though not specifically used in this heuristic.

    Returns:
        ActionOperator: An operator with all actions set to 0, indicating no charging.
        dict: An empty dictionary since no algorithm data is updated in this heuristic.
    """
    fleet_size = len(state_data["ride_lead_time"])
    actions = [0] * fleet_size  # Default all actions to 0, meaning no vehicle will charge

    # Ensure constraints are met: vehicles on a ride cannot charge, and fully charged vehicles should not charge.
    for i in range(fleet_size):
        if state_data["ride_lead_time"][i] >= 2 or state_data["battery_soc"][i] >= 1:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > global_data["total_chargers"]:
        actions = [0] * fleet_size  # Set all to 0 if the constraint is violated

    operator = ActionOperator(actions)
    return operator, {}