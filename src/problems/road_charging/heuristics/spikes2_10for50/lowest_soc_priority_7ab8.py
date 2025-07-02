from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_7ab8(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for dynamically prioritizing EVs based on operational status and battery SoC.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - order_price (list[float]): Payment received per minute for a ride, indexed by time step.
            - charging_price (list[float]): Charging cost per kilowatt-hour, indexed by time step.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): Operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.
            - time_to_next_availability (list[int]): Lead time until the fleet becomes available.
            - battery_soc (list[float]): Battery state of charge in percentage for each EV.
        
        kwargs: Hyper-parameters for the algorithm, including:
            - base_critical_SoC_threshold (float, optional, default=0.15): The base SoC threshold for prioritizing charging.

    Returns:
        ActionOperator with charging actions for EVs based on their operational status and dynamically adjusted SoC threshold.
        An empty dictionary as no algorithm data needs to be updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    order_price = global_data["order_price"]
    charging_price = global_data["charging_price"]
    base_critical_SoC_threshold = kwargs.get("base_critical_SoC_threshold", 0.15)
    
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data.get("current_step", 0)

    # Dynamically adjust critical SoC threshold based on real-time prices
    price_factor = order_price[current_step] / (charging_price[current_step] + 0.01)
    critical_SoC_threshold = base_critical_SoC_threshold * price_factor

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Identify EVs eligible for charging based on operational status and SoC
    chargeable_evs = [
        i for i in range(fleet_size)
        if (operational_status[i] == 1 and time_to_next_availability[i] == 0 and battery_soc[i] < critical_SoC_threshold) or
           (operational_status[i] == 0 and battery_soc[i] < critical_SoC_threshold)
    ]

    # If no EVs are eligible to charge, return zero actions
    if not chargeable_evs:
        return ActionOperator(actions), {}

    # Sort chargeable EVs by their state of charge (SoC) in ascending order
    chargeable_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in chargeable_evs[:total_chargers]:
        actions[i] = 1

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, {}