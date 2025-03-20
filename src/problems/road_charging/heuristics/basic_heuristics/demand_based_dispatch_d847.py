from src.problems.base.mdp_components import Solution, ActionOperator

def demand_based_dispatch_d847(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ DemandBasedDispatch Heuristic: Allocates EVs to remain available or go to charge based on customer demand.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Total number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total available charging stations.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "current_step" (int): Current time step index.
            - "operational_status" (list[int]): Status of each EV (0: idle, 1: serving, 2: charging).
            - "time_to_next_availability" (list[int]): Time until each EV is available.
            - "battery_soc" (list[float]): Battery state of charge for each EV.

        algorithm_data (dict): This algorithm does not use specific algorithm data, but it can be included if needed for extensions.

        get_state_data_function (callable): Function to get state data for a potential new solution. Not used in this basic implementation.

        **kwargs: Optional hyper-parameters for this heuristic:
            - "low_soc_threshold" (float, default=0.2): Threshold below which EVs are prioritized for charging.
            - "high_demand_threshold" (int, default=5): Threshold of customer arrivals to consider it a high demand period.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Empty dictionary since no algorithm-specific updates are made.
    """
    
    low_soc_threshold = kwargs.get('low_soc_threshold', 0.2)
    high_demand_threshold = kwargs.get('high_demand_threshold', 5)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Calculate demand at the current step
    current_demand = customer_arrivals[current_step]

    # Determine the number of EVs that can charge based on charger availability
    available_chargers = total_chargers

    # Identify EVs that should prioritize charging
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:  # EV is on a ride
            actions[i] = 0  # Must remain available
        elif battery_soc[i] < low_soc_threshold and available_chargers > 0:
            actions[i] = 1  # Set to charge
            available_chargers -= 1

    # If high demand, prioritize availability
    if current_demand > high_demand_threshold:
        # Reset actions for EVs with lower priority to charge if demand is high
        for i in range(fleet_size):
            if actions[i] == 1 and available_chargers < total_chargers:
                actions[i] = 0  # Reset to remain available
                available_chargers += 1

    # Ensure the sum of actions does not exceed total chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    return ActionOperator(actions), {}