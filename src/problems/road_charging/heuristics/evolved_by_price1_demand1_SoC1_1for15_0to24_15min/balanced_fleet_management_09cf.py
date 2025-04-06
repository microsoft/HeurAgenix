from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balanced_fleet_management_09cf(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Balanced Fleet Management Heuristic Algorithm with Dynamic SoC Threshold Adjustment.

    This algorithm dynamically manages EV charging decisions based on real-time demand trends and battery states to optimize fleet operations.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - max_time_steps (int): Maximum number of time steps.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Current time step index.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV.
        kwargs: Hyper-parameters necessary for the algorithm:
            - base_charging_threshold (float): Base SoC threshold for prioritizing charging, default is 0.5.
            - demand_trend_window (int): Window size for calculating rolling average of customer arrivals, default is 5.

    Returns:
        ActionOperator: Operator to execute the fleet management strategy.
        dict: Empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Set default hyper-parameters
    base_charging_threshold = kwargs.get("base_charging_threshold", 0.5)
    demand_trend_window = kwargs.get("demand_trend_window", 5)

    # Calculate rolling average of customer arrivals to identify demand trends
    if current_step >= demand_trend_window:
        recent_demand_trend = np.mean(customer_arrivals[max(0, current_step - demand_trend_window):current_step])
    else:
        recent_demand_trend = np.mean(customer_arrivals[:current_step + 1])

    # Adjust charging threshold based on demand trend
    charging_priority_threshold = base_charging_threshold
    if recent_demand_trend > np.mean(customer_arrivals):
        charging_priority_threshold += 0.1  # Increase threshold during sustained high demand

    # Initialize actions for each EV to zero (remain available).
    actions = [0] * fleet_size

    # Prioritize charging for idle EVs with low SoC based on dynamic threshold adjustment.
    charging_candidates = [i for i in range(fleet_size) if operational_status[i] == 0 and battery_soc[i] < charging_priority_threshold]

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