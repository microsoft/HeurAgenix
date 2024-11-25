from src.problems.base.mdp_components import *

def least_charging_needed_e88a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Least Charging Needed heuristic for the road_charging problem.
    
    This heuristic prioritizes EVs that require the least amount of charging to reach their target state of charge. It assesses all unscheduled EVs waiting to charge and selects the one with the least remaining charging requirement.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
            - "total_chargers" (int): The number of available chargers.
            - "target_soc" (list[int]): The target state of charge for each EV.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[int]): The current state of charge for each EV.
            - "ride_lead_time" (list[int]): The remaining ride time if an EV is currently on a ride.
            - "charging_lead_time" (list[int]): The time an EV has been charging.

    Returns:
        ActionOperator: The operator that schedules charging actions for each EV based on least remaining charging needed.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    target_soc = global_data["target_soc"]
    
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]
    
    actions = [0] * fleet_size  # Initialize all actions to 0 (no charging)
    
    available_chargers = total_chargers - sum(charging_lead_time)
    
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    min_charging_needed = float('inf')
    best_ev = None

    # Calculate remaining charging needed for each EV
    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            continue  # Do not charge if EV is on a ride

        remaining_charge = target_soc[i] - battery_soc[i]
        if remaining_charge < min_charging_needed:
            min_charging_needed = remaining_charge
            best_ev = i

    if best_ev is not None:
        actions[best_ev] = 1
        return ActionOperator(actions), {}

    # Return default action if no improvement was found
    return ActionOperator(actions), {}