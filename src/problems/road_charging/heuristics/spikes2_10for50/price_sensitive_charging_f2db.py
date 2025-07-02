from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_f2db(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Enhanced heuristic algorithm with dynamic adjustment for charging sessions based on historical data trends and predictive analytics.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
            - "current_step" (int): The index of the current time step.
        kwargs: Hyper-parameters used in the algorithm.
            - "soc_threshold" (float, optional): Dynamically adjusted threshold for prioritizing EVs for charging.
            - "charging_price_threshold" (float, optional): Dynamically adjusted average charging price threshold.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    
    # Dynamic hyper-parameters adjustment using historical and real-time data
    historical_soc_threshold = kwargs.get("soc_threshold", 0.8)
    historical_charging_price_threshold = kwargs.get("charging_price_threshold", 0.30)

    # Example of dynamic adjustment based on historical trends (can be improved with actual analytics)
    soc_threshold = np.median(battery_soc) * 0.9 if current_step % 10 == 0 else historical_soc_threshold
    charging_price_threshold = np.mean(charging_price[:current_step]) if current_step % 10 == 0 else historical_charging_price_threshold

    # Initialize actions with zeros
    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size
    
    # Check if charging_price data is available
    if not charging_price:
        # If no charging price data, return an action operator with all zeros
        return ActionOperator(actions), {}

    # Prioritize EVs for charging based on dynamically adjusted thresholds
    ev_indices = np.argsort(battery_soc)
    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            # EV is currently on a ride, cannot charge
            actions[i] = 0
        elif chargers_used < total_chargers and battery_soc[i] < soc_threshold and charging_price[current_step] <= charging_price_threshold:
            # Assign charging action if chargers are available, SoC is below threshold, and current price is optimal
            actions[i] = 1
            chargers_used += 1
        else:
            # No more chargers available or conditions not met, remaining EVs stay idle
            actions[i] = 0

    # Ensure the sum of actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}