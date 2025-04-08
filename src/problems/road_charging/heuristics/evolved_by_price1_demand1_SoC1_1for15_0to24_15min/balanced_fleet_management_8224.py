from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balanced_fleet_management_8224(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Balanced Fleet Management Heuristic Algorithm with Dynamic SoC Threshold Adjustment.

    This algorithm dynamically manages EV charging decisions based on real-time demand trends, battery states, fleet conditions, and historical data to optimize operations. It includes a smoothing mechanism for scaling factors to reduce sensitivity to short-term fluctuations.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - max_time_steps (int): Total number of time steps.
            - order_price (list[float]): List of prices per order at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Current time step index.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV.
        kwargs: Hyper-parameters necessary for the algorithm:
            - demand_trend_window (int): Window size for calculating rolling average of customer arrivals, default is 5.
            - scaling_factor (float): Scaling factor for adjusting charge_lb and charge_ub based on rate of change, default is 0.05.
            - smoothing_alpha (float): Smoothing factor for exponential moving average, default is 0.3.

    Returns:
        ActionOperator: Operator to execute the fleet management strategy.
        dict: Empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Set default hyper-parameters
    demand_trend_window = kwargs.get("demand_trend_window", 5)
    scaling_factor = kwargs.get("scaling_factor", 0.05)
    smoothing_alpha = kwargs.get("smoothing_alpha", 0.3)

    # Calculate rolling average of customer arrivals to identify demand trends
    if current_step >= demand_trend_window:
        recent_demand_trend = np.mean(customer_arrivals[max(0, current_step - demand_trend_window):current_step])
        previous_demand_trend = np.mean(customer_arrivals[max(0, current_step - demand_trend_window - 1):current_step - 1])
    else:
        recent_demand_trend = np.mean(customer_arrivals[:current_step + 1])
        previous_demand_trend = recent_demand_trend

    # Calculate peak customer arrivals
    peak_customer_arrivals = max(customer_arrivals) if customer_arrivals else 1

    # Dynamic adjustment factors based on instance characteristics and real-time conditions
    average_order_price = np.mean(order_price) if order_price else 0.0
    change_rate = (recent_demand_trend - previous_demand_trend) / previous_demand_trend if previous_demand_trend != 0 else 0
    smoothed_change_rate = smoothing_alpha * change_rate + (1 - smoothing_alpha) * algorithm_data.get('previous_change_rate', 0)
    charge_lb = (0.55 + (average_order_price / peak_customer_arrivals) * 0.1) * (1 + scaling_factor * smoothed_change_rate)
    charge_ub = (0.60 + (recent_demand_trend / peak_customer_arrivals) * 0.1) * (1 + scaling_factor * smoothed_change_rate)

    # Update algorithm_data with the smoothed change rate for next iteration
    algorithm_data['previous_change_rate'] = smoothed_change_rate

    # Initialize actions for each EV to zero (remain available).
    actions = [0] * fleet_size

    # Determine actions for each EV
    chargers_used = 0
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            if chargers_used < total_chargers:
                actions[i] = 1
                chargers_used += 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Create and return the ActionOperator with the new actions.
    operator = ActionOperator(actions)
    return operator, algorithm_data