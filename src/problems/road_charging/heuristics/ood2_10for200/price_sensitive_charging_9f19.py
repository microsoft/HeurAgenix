from src.problems.base.mdp_components import ActionOperator
import numpy as np

def price_sensitive_charging_9f19(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, charge_lb=0.50, charge_ub=0.75, fleet_size_threshold=5, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for optimizing EV charging with strategic SoC threshold adjustments.

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
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used in this algorithm.
        charge_lb (float): Lower bound for the battery state of charge to initiate charging.
        charge_ub (float): Upper bound for the battery state of charge to stop charging.
        fleet_size_threshold (int): Threshold for fleet size to apply prioritization logic.

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
        return ActionOperator([0] * fleet_size), {}

    average_customer_arrivals = np.mean(customer_arrivals) if customer_arrivals else 0

    # Relax charge_lb when customer arrivals are above average
    if customer_arrivals[current_step] > average_customer_arrivals:
        charge_lb -= 0.05

    # Check fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers
    if fleet_to_charger_ratio > 5.0:
        charge_lb += 0.05

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

    # Ensure valid action sum does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}