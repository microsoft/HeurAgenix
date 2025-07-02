from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_69a0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, average_charging_price: float = 0.30, early_charging_bias: float = 1.5, fleet_to_charger_ratio_threshold: float = 3.0) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to optimize EV charging sessions by prioritizing low charging prices and early charging opportunities.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - total_chargers (int): The maximum number of available chargers.
            - charging_price (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory for EVs.
            - time_to_next_availability (list[int]): Lead time until the fleet becomes available.
            - battery_soc (list[float]): Battery state of charge in percentage.
            - current_step (int): The index of the current time step.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used in this algorithm.
        average_charging_price (float): The average charging price to prioritize charging at lower rates, default is 0.30.
        early_charging_bias (float): A bias factor to favor early charging in competitive scenarios, default is 1.5.
        fleet_to_charger_ratio_threshold (float): The threshold for fleet-to-charger ratio to apply certain logic, default is 3.0.

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

    # Determine if conditions are favorable for charging
    if charging_price[current_step] <= average_charging_price * early_charging_bias and fleet_size / total_chargers >= fleet_to_charger_ratio_threshold:
        # Sort EVs by battery SoC in ascending order (prioritizing lower SoC for charging)
        ev_indices = np.argsort(battery_soc)
        
        # Attempt to assign charging actions to EVs with lower battery SoC and availability
        chargers_used = 0
        for i in ev_indices:
            if time_to_next_availability[i] == 0 and chargers_used < total_chargers:
                # Assign charging action only if the EV is available and chargers are available
                actions[i] = 1
                chargers_used += 1
            if chargers_used >= total_chargers:
                break
    
    # Ensure the sum of actions does not exceed the number of available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size
    
    return ActionOperator(actions), {}