from src.problems.base.mdp_components import ActionOperator
import numpy as np

def price_sensitive_charging_b853(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, fleet_size_threshold=5, soc_threshold=0.5, price_factor=0.9, charger_ratio_threshold=5, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for optimizing EV charging based on fleet size, urgency, and price sensitivity.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "max_time_steps" (int): Maximum number of time steps in the scenario.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): Operational status of EVs (0: idle, 1: serving a trip, 2: charging).
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.
        fleet_size_threshold (int): Threshold for fleet size to apply prioritization logic. Defaults to 5.
        soc_threshold (float): SoC threshold to prioritize charging. Defaults to 0.5.
        price_factor (float): Factor to determine significant charging price drop. Defaults to 0.9.
        charger_ratio_threshold (int): Threshold for fleet-to-charger ratio to apply prioritization logic. Defaults to 5.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    total_chargers = global_data["total_chargers"]
    charging_price = global_data.get("charging_price", [])
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"].tolist()  # Convert numpy array to list for counting
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    
    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size
    
    # If charging_price data is unavailable or current_step exceeds its length, return all zeros
    if not charging_price or current_step >= len(charging_price):
        return ActionOperator(actions), {}
    
    # Calculate average charging price for comparison
    average_charging_price = np.mean(charging_price) if charging_price else 0

    # Sort EVs by battery SoC in ascending order
    ev_indices = np.argsort(battery_soc)
    
    # Prioritize EVs for charging based on SoC and charging price conditions
    chargers_used = 0
    for i in ev_indices:
        if operational_status[i] == 0 and time_to_next_availability[i] == 0 and battery_soc[i] < soc_threshold:
            if (fleet_size / total_chargers > charger_ratio_threshold) or (charging_price[current_step] < average_charging_price * price_factor):
                if chargers_used < total_chargers:
                    actions[i] = 1
                    chargers_used += 1
            if chargers_used >= total_chargers:
                break
    
    # Implement logic to prioritize lowest SoC when many EVs are idle
    if operational_status.count(0) > fleet_size_threshold:
        for i in ev_indices:
            if operational_status[i] == 0 and time_to_next_availability[i] == 0:
                if chargers_used < total_chargers:
                    actions[i] = 1
                    chargers_used += 1
            if chargers_used >= total_chargers:
                break
    
    # Ensure valid action sum does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}