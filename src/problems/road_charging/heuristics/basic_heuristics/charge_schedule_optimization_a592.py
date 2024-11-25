from src.problems.base.mdp_components import *
import numpy as np

def charge_schedule_optimization_a592(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Charge Schedule Optimization heuristic for the road_charging problem.
    
    This algorithm evaluates potential improvements in the charging schedule by examining pairs of charging actions and determining if swapping or adjusting them can yield a lower total cost or higher overall benefit. The algorithm iteratively identifies and applies the best modifications to the charging schedule.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
            - "total_chargers" (int): The number of available chargers.
            - "charging_price" (list[float]): The price per kWh at each time step.
            - "order_price" (list[float]): The earnings per time step for each EV.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[int]): The state of charge for each EV.
            - "ride_lead_time" (list[int]): The remaining ride time if an EV is currently on a ride.
            - "charging_lead_time" (list[int]): The time an EV has been charging.

    Returns:
        ActionOperator: The operator that schedules charging actions for each EV based on optimized schedule.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    order_price = global_data["order_price"]
    
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]
    
    actions = [0] * fleet_size  # Initialize all actions to 0 (no charging)
    
    available_chargers = total_chargers - sum(charging_lead_time)
    
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    best_delta = 0
    best_action = None

    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            continue  # Do not charge if EV is on a ride

        for j in range(fleet_size):
            if i == j or ride_lead_time[j] > 0:
                continue  # Skip same EV or if the other EV is on a ride

            # Calculate potential cost savings of swapping charging actions
            current_cost_i = charging_price[i] * battery_soc[i]
            current_cost_j = charging_price[j] * battery_soc[j]
            potential_cost_i = charging_price[j] * battery_soc[i]
            potential_cost_j = charging_price[i] * battery_soc[j]
            
            delta = (current_cost_i + current_cost_j) - (potential_cost_i + potential_cost_j)

            if delta > best_delta:
                best_delta = delta
                best_action = (i, j)
    
    if best_action:
        i, j = best_action
        actions[i], actions[j] = 1, 0
        return ActionOperator(actions), {}

    # Return default action if no improvement was found
    return ActionOperator(actions), {}