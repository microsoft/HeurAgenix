from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_f76c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for prioritizing EVs with a recalibrated dynamic SoC threshold using feedback and prediction.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "min_SoC" (float): The safety battery SoC threshold.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "battery_soc" (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - "time_to_next_availability" (list[int]): A 1D array indicating the lead time until the fleet becomes available.
            - "operational_status" (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.

    Returns:
        ActionOperator to assign charging actions to EVs based on their SoC and predicted demand.
        An empty dictionary as no algorithm data needs to be updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    min_SoC = global_data["min_SoC"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]

    # Recalibrate the dynamic threshold using feedback and prediction
    average_customer_arrivals = kwargs.get("average_customer_arrivals", 11.60)
    peak_customer_arrivals = kwargs.get("peak_customer_arrivals", 19)
    historical_soc_level = kwargs.get("historical_soc_level", np.mean(battery_soc))
    idle_frequency = kwargs.get("idle_frequency", np.mean([1 if status == 0 else 0 for status in operational_status]))

    # Feedback loop: Adjust weights based on current performance data
    soc_weight = kwargs.get("soc_weight", 0.5)
    idle_weight = kwargs.get("idle_weight", 0.5)
    
    # More sophisticated prediction model placeholder
    predicted_demand_factor = (peak_customer_arrivals - average_customer_arrivals) / peak_customer_arrivals

    # Dynamic threshold recalibration
    dynamic_threshold = min_SoC + predicted_demand_factor * idle_weight + (1 - historical_soc_level) * soc_weight

    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Filter EVs that are idle and have SoC below dynamic threshold
    chargeable_evs = [
        i for i in range(fleet_size)
        if operational_status[i] == 0 and time_to_next_availability[i] == 0 and battery_soc[i] < dynamic_threshold
    ]

    # If no EVs are available to charge, return zero actions
    if not chargeable_evs:
        return ActionOperator(actions), {}

    # Sort chargeable EVs by their state of charge (SoC) in ascending order,
    # prioritizing those closest to dynamic_threshold
    chargeable_evs.sort(key=lambda i: abs(battery_soc[i] - dynamic_threshold))

    # Limit to available chargers
    chargeable_evs = chargeable_evs[:total_chargers]

    # Assign charging actions
    for i in chargeable_evs:
        actions[i] = 1

    # Ensure hard constraints are met
    if sum(actions) > total_chargers:
        for i in range(len(actions)):
            if actions[i] == 1:
                actions[i] = 0
                if sum(actions) <= total_chargers:
                    break

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, {}