from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_6e8d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm with refined dynamic threshold adjustment and historical demand prediction.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
        algorithm_data (dict): Not used in this heuristic.
        get_state_data_function (callable): Not used in this heuristic.
        base_threshold (float, optional): Base battery SoC threshold for charging. Default is 0.7.
        demand_tiers (list[float], optional): Tiered adjustments for dynamic threshold based on demand intensity. Default is [0.05, 0.1, 0.15].

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data.get("current_step", 0)
    base_threshold = kwargs.get("base_threshold", 0.7)
    demand_tiers = kwargs.get("demand_tiers", [0.05, 0.1, 0.15])

    # Determine demand intensity tier based on historical and current demand
    average_demand = np.mean(customer_arrivals)
    current_demand = customer_arrivals[current_step] if current_step < len(customer_arrivals) else average_demand
    if current_demand > 1.5 * average_demand:
        demand_adjustment = demand_tiers[2]
    elif current_demand > average_demand:
        demand_adjustment = demand_tiers[1]
    else:
        demand_adjustment = demand_tiers[0]

    dynamic_threshold = base_threshold - demand_adjustment

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Sort idle EVs by battery SoC
    idle_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] == 0]
    sorted_idle_evs = sorted(idle_evs, key=lambda i: battery_soc[i])

    # Determine actions based on battery SoC and operational status
    for i in sorted_idle_evs:
        # If the EV is idle and its SoC is below the dynamic threshold, consider charging
        if battery_soc[i] < dynamic_threshold:
            actions[i] = 1

    # Ensure the number of charging actions does not exceed the total chargers
    if sum(actions) > total_chargers:
        for idx in sorted_idle_evs:
            if sum(actions) > total_chargers:
                actions[idx] = 0

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, {}