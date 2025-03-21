from src.problems.base.mdp_components import Solution, ActionOperator

def balanced_fleet_management_6981(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Balanced Fleet Management Heuristic Algorithm.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - max_time_steps (int): Maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV.
        algorithm_data (dict): No specific algorithm data is required for this heuristic.
        get_state_data_function (callable): Function to retrieve state data for a new solution. Not used in this heuristic.
        kwargs (dict): No hyper-parameters are used in this heuristic.

    Returns:
        ActionOperator: Operator to execute the balanced fleet management strategy.
        dict: Empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions for each EV to zero (remain available).
    actions = [0] * fleet_size

    # Identify EVs that are idle and have battery SoC below a threshold.
    charging_candidates = [i for i in range(fleet_size) if operational_status[i] == 0 and battery_soc[i] < 0.5]

    # Limit the number of EVs going to charge by the number of available chargers.
    chargers_used = 0
    for i in charging_candidates:
        if chargers_used < total_chargers:
            actions[i] = 1  # Set action to charge.
            chargers_used += 1

    # Ensure EVs on a ride continue to remain available.
    for i in range(fleet_size):
        if time_to_next_availability[i] > 0:
            actions[i] = 0

    # Create and return the ActionOperator with the new actions.
    operator = ActionOperator(actions)
    return operator, {}