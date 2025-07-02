from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_7167(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to schedule charging sessions during periods of lower charging prices and when battery SoC is below a threshold.

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
            - "soc_threshold" (float, default=0.8): The threshold below which EVs are prioritized for charging.
            - "average_charging_price" (float, default=0.30): The average charging price used to determine low-cost charging opportunities.

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
    
    # Hyper-parameters
    soc_threshold = kwargs.get("soc_threshold", 0.8)
    average_charging_price = kwargs.get("average_charging_price", 0.30)

    # Initialize actions with zeros
    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size
    
    # Check if charging_price data is available
    if not charging_price:
        # If no charging price data, return an action operator with all zeros
        return ActionOperator(actions), {}

    # Prioritize EVs for charging based on SoC and charging price
    ev_indices = np.argsort(battery_soc)
    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            # EV is currently on a ride, cannot charge
            actions[i] = 0
        elif chargers_used < total_chargers and battery_soc[i] < soc_threshold and charging_price[current_step] <= average_charging_price:
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