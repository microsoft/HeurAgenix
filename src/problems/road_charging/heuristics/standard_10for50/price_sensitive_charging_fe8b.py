from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_fe8b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to dynamically schedule EV charging sessions based on pricing, battery SoC, and demand conditions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - total_chargers (int): The maximum number of available chargers.
            - charging_price (list[float]): A list representing the charging price at each time step.
            - min_SoC (float): The minimum battery SoC threshold for dispatch eligibility.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - order_price (list[float]): A list representing the price received per order at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory for EVs.
            - time_to_next_availability (list[int]): Lead time until the fleet becomes available.
            - battery_soc (list[float]): Battery state of charge in percentage.
            - current_step (int): The index of the current time step.
        get_state_data_function (callable): Function to get state data for new solution without modifying the original.
        kwargs: Hyper parameters for controlling charging behavior:
            - base_aggressive_threshold (float): Default 0.15. Base threshold for aggressive charging.
            - dynamic_adjustment_factor (float): Default 0.05. Factor to adjust threshold dynamically based on fleet_to_charger_ratio.
            - demand_sensitivity (float): Default 0.01. Sensitivity to peak customer arrivals and average order price.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    min_SoC = global_data["min_SoC"]
    fleet_size = global_data["fleet_size"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]
    current_step = state_data["current_step"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    
    # Calculate fleet_to_charger_ratio
    fleet_to_charger_ratio = fleet_size / total_chargers if total_chargers > 0 else float('inf')
    
    # Hyper-parameters
    base_aggressive_threshold = kwargs.get("base_aggressive_threshold", 0.15)
    dynamic_adjustment_factor = kwargs.get("dynamic_adjustment_factor", 0.05)
    demand_sensitivity = kwargs.get("demand_sensitivity", 0.01)
    
    # Calculate dynamic aggressive threshold
    aggressive_threshold = base_aggressive_threshold - (dynamic_adjustment_factor * (fleet_to_charger_ratio - 1))
    
    # Calculate peak customer arrivals
    peak_customer_arrivals = max(customer_arrivals) if customer_arrivals else 0
    
    # Calculate average order price
    average_order_price = np.mean(order_price) if order_price else 0
    
    # Adjust threshold based on peak demand and pricing
    demand_adjustment = demand_sensitivity * (peak_customer_arrivals * average_order_price)
    aggressive_threshold -= demand_adjustment

    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Check if charging_price data is available
    if not charging_price:
        # If no charging price data, return an action operator with all zeros
        return ActionOperator(actions), {}

    # Calculate average historical charging price
    historical_avg_price = np.mean(charging_price[:current_step]) if current_step > 0 else charging_price[0]

    # Sort EVs by battery SoC in ascending order (prioritizing lower SoC for charging) and by idle status
    ev_indices = np.argsort(battery_soc + np.array(time_to_next_availability) * 10)  # Amplify effect of being idle

    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            # EV is currently on a ride, cannot charge
            actions[i] = 0
        elif (battery_soc[i] <= min_SoC or charging_price[current_step] < historical_avg_price) and chargers_used < total_chargers:
            # Charge if SoC is near threshold or current price is favorable
            actions[i] = 1
            chargers_used += 1
        elif fleet_to_charger_ratio >= 5 and charging_price[current_step] <= aggressive_threshold:
            # More aggressive charging when fleet_to_charger_ratio is high
            actions[i] = 1
            chargers_used += 1
        else:
            # Stay idle if no suitable conditions met
            actions[i] = 0

    return ActionOperator(actions), {}