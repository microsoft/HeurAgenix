from src.problems.base.mdp_components import ActionOperator
import numpy as np

def greedy_charging_b79f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm with dynamic scaling of hyper-parameters based on fleet-to-charger ratio and real-time conditions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
        kwargs: 
            - base_threshold (float, default=0.7): The initial battery SoC threshold below which an EV should prioritize charging.
            - base_priority_factor (float, default=0.1): The initial factor to weigh the urgency of charging based on SoC.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]

    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    base_threshold = kwargs.get("base_threshold", 0.9)
    base_priority_factor = kwargs.get("base_priority_factor", 0.1)

    # Calculate dynamic parameters based on real-time conditions and fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers
    average_customer_arrivals = np.mean(customer_arrivals[max(0, current_step - 5):current_step + 1])
    
    dynamic_priority_factor = base_priority_factor + (average_customer_arrivals / 100) * (fleet_to_charger_ratio / 10)
    dynamic_threshold = base_threshold - (average_customer_arrivals / 1000) * (fleet_to_charger_ratio / 10)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Enhanced prioritization logic for larger fleets and high fleet-to-charger ratios
    if fleet_size > 5 and fleet_to_charger_ratio > 5:
        # Identify the EV with the lowest SoC
        min_soc_index = np.argmin(battery_soc)
        # Prioritize charging for the EV with the lowest SoC
        if operational_status[min_soc_index] == 0 and time_to_next_availability[min_soc_index] == 0:
            actions[min_soc_index] = 1

    # Default heuristic logic with dynamic adjustment
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        elif operational_status[i] == 0 and battery_soc[i] < dynamic_threshold:
            adjusted_threshold = dynamic_threshold - dynamic_priority_factor * (battery_soc[i] / fleet_size)
            if battery_soc[i] < adjusted_threshold:
                actions[i] = 1

    # Ensure the number of charging actions does not exceed the total chargers
    if sum(actions) > total_chargers:
        # Select EVs with the lowest SoC to charge
        soc_indices = np.argsort(battery_soc)
        chargers_used = 0
        for idx in soc_indices:
            if chargers_used < total_chargers and actions[idx] == 1:
                chargers_used += 1
            else:
                actions[idx] = 0

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, {}