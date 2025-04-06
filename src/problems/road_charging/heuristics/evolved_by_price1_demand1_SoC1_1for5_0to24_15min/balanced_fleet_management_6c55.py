from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balanced_fleet_management_6c55(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Dynamic Balanced Fleet Management Heuristic Algorithm.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - charging_price (list[float]): Charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV.
            - current_step (int): Current time step index.
        algorithm_data (dict): May contain historical performance data for threshold adjustment.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator: Operator to execute the dynamic balanced fleet management strategy.
        dict: Updated algorithm data with historical performance metrics for future adjustments.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    charging_price = global_data["charging_price"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Calculate average customer arrivals and charging price at the current step
    avg_customer_arrivals = np.mean(customer_arrivals[max(0, current_step - 5):current_step + 1])
    avg_charging_price = np.mean(charging_price[max(0, current_step - 5):current_step + 1])

    # Dynamically adjust charging priority threshold based on real-time factors
    charging_priority_threshold = kwargs.get("base_threshold", 0.5)
    charging_priority_threshold *= (avg_customer_arrivals / fleet_size) * (1 / (avg_charging_price + 0.01))

    # Initialize actions for each EV to zero (remain available).
    actions = [0] * fleet_size

    # Identify EVs that are idle and have battery SoC below the threshold.
    charging_candidates = [
        i for i in range(fleet_size)
        if operational_status[i] == 0 and battery_soc[i] < charging_priority_threshold
    ]

    # Sort the charging candidates by their SoC in ascending order to prioritize lower SoC
    charging_candidates.sort(key=lambda i: battery_soc[i])

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

    # Ensure the sum of actions does not exceed the number of chargers available
    if sum(actions) > total_chargers:
        # Reset excess charging actions to 0 to comply with charger constraints
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[total_chargers:]:
            actions[index] = 0

    # If no EV is selected to charge, check if any idle EV is available and select the one with the lowest SoC
    if sum(actions) == 0:
        idle_evs = [i for i in range(fleet_size) if operational_status[i] == 0]
        if idle_evs:
            min_soc_ev = min(idle_evs, key=lambda i: battery_soc[i])
            actions[min_soc_ev] = 1

    # Create and return the ActionOperator with the new actions.
    operator = ActionOperator(actions)
    return operator, {}