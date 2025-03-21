from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_65e1(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to schedule charging sessions during periods of lower charging prices.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
            - "current_step" (int): The index of the current time step.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used in this algorithm.
        kwargs: Not used in this algorithm.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    
    # Initialize actions with zeros
    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size
    
    # Check if charging_price data is available
    if not charging_price:
        # If no charging price data, return an action operator with all zeros
        return ActionOperator(actions), {}

    # Find the minimum charging price across future time steps to decide on charging
    min_future_price = min(charging_price[current_step:])
    
    # Sort EVs by battery SoC in ascending order (prioritizing lower SoC for charging)
    ev_indices = np.argsort(battery_soc)

    # Attempt to assign charging actions to EVs with lower battery SoC and availability
    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            # EV is currently on a ride, cannot charge
            actions[i] = 0
        elif chargers_used < total_chargers and charging_price[current_step] <= min_future_price:
            # Assign charging action only if chargers are available and current price is not higher than any future prices
            actions[i] = 1
            chargers_used += 1
        else:
            # No more chargers available or current price is not optimal, remaining EVs stay idle
            break

    return ActionOperator(actions), {}