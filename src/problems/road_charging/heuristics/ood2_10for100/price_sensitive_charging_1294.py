from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_1294(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for scheduling EV charging sessions with dynamic SoC threshold adjustment.

    Args:
        global_data (dict): The global data dict containing the necessary data for the algorithm:
            - "total_chargers" (int): Maximum number of available chargers.
            - "charging_price" (list[float]): List of charging prices at each time step.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.
            - "charging_rate" (list[float]): Charging rate per time step for each vehicle.
        state_data (dict): The state dictionary containing the current state information:
            - "current_solution" (Solution): Current action trajectory for EVs.
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
            - "current_step" (int): Index of the current time step.

    Returns:
        ActionOperator: Operator defining new charging actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """

    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data.get("charging_price", [])
    customer_arrivals = global_data.get("customer_arrivals", [])
    charging_rate = global_data.get("charging_rate", [])
    current_step = state_data["current_step"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Calculate dynamic SoC threshold based on peak customer arrivals
    if customer_arrivals:
        peak_customer_arrivals = np.max(customer_arrivals)
        current_customer_arrivals = customer_arrivals[current_step]
        dynamic_soc_threshold = np.clip(0.75 - (current_customer_arrivals / peak_customer_arrivals) * 0.25, 0.5, 0.75)
    else:
        dynamic_soc_threshold = 0.75

    # Initialize actions with zeros
    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size

    # Ensure charging_price data is available
    if not charging_price:
        return ActionOperator(actions), {}

    # Determine if current period is low demand and low price
    is_low_demand = customer_arrivals[current_step] < np.mean(customer_arrivals)
    is_low_price = charging_price[current_step] < np.mean(charging_price)

    # Adjust charging strategy based on demand and price
    if np.mean(battery_soc) < dynamic_soc_threshold or (is_low_demand and is_low_price):
        # Prioritize charging for the EV with the lowest SoC
        ev_indices = np.argsort(battery_soc)
        chargers_used = 0
        for i in ev_indices:
            if time_to_next_availability[i] == 0 and chargers_used < total_chargers:
                actions[i] = 1
                chargers_used += 1
            # Stop assigning charging actions if all chargers are used
            if chargers_used >= total_chargers:
                break

    # Ensure the sum of actions does not exceed the maximum number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}