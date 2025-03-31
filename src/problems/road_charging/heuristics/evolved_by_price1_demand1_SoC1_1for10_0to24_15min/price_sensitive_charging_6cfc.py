from src.problems.base.mdp_components import ActionOperator
import numpy as np

def price_sensitive_charging_6cfc(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, fleet_size_threshold=5, base_soc_threshold=0.68, price_factor=0.9, charger_ratio_threshold=5, peak_customer_arrival_threshold=8, price_threshold=0.35, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for optimizing EV charging with granular dynamic SoC adjustment.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
            - "max_time_steps" (int): Maximum number of time steps in the scenario.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): Operational status of EVs (0: idle, 1: serving a trip, 2: charging).
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
        fleet_size_threshold (int): Threshold for fleet size to apply prioritization logic. Defaults to 5.
        base_soc_threshold (float): Base SoC threshold to prioritize charging. Defaults to 0.68.
        price_factor (float): Factor to determine significant charging price drop. Defaults to 0.9.
        charger_ratio_threshold (int): Threshold for fleet-to-charger ratio to apply prioritization logic. Defaults to 5.
        peak_customer_arrival_threshold (int): Threshold for peak customer arrival to apply additional logic. Defaults to 8.
        price_threshold (float): Charging price threshold to apply prioritization logic. Defaults to 0.35.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    total_chargers = global_data["total_chargers"]
    charging_price = global_data.get("charging_price", [])
    customer_arrivals = global_data.get("customer_arrivals", [])
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"].tolist()  # Convert numpy array to list for counting
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size

    # Ensure charging_price and customer_arrivals data is available and current_step is valid
    if not charging_price or current_step >= len(charging_price) or not customer_arrivals or current_step >= len(customer_arrivals):
        return ActionOperator(actions), {}

    # Calculate average charging price for comparison
    average_charging_price = np.mean(charging_price) if charging_price else 0
    average_customer_arrivals = np.mean(customer_arrivals) if customer_arrivals else 0
    peak_customer_arrivals = max(customer_arrivals) if customer_arrivals else 0

    # Calculate dynamic SoC threshold adjustments
    arrival_ratio = (customer_arrivals[current_step] / average_customer_arrivals) if average_customer_arrivals else 1
    dynamic_soc_threshold = base_soc_threshold

    if arrival_ratio >= 1.5:
        dynamic_soc_threshold += 0.10  # Increase SoC threshold more during significantly high demand
    elif arrival_ratio >= 1.2:
        dynamic_soc_threshold += 0.05  # Moderate increase during moderately high demand

    if charging_price[current_step] < price_threshold:
        dynamic_soc_threshold += 0.05  # Slightly increase SoC threshold during low price

    # Sort EVs by battery SoC in ascending order
    ev_indices = np.argsort(battery_soc)

    # Prioritize EVs for charging based on dynamic SoC threshold and charging conditions
    chargers_used = 0
    for i in ev_indices:
        if operational_status[i] == 0 and time_to_next_availability[i] == 0 and battery_soc[i] < dynamic_soc_threshold:
            if (fleet_size / total_chargers > charger_ratio_threshold) or (charging_price[current_step] < average_charging_price * price_factor) or (customer_arrivals[current_step] >= peak_customer_arrival_threshold) or (charging_price[current_step] < price_threshold):
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