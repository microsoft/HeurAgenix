from src.problems.base.mdp_components import *

def first_come_first_charged_257c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ First Come First Charged heuristic for the road_charging problem.
    
    This heuristic schedules EVs to charge in the order they request charging, without considering other factors such as state of charge or cost. It maintains a queue of EVs waiting to charge and assigns charging slots to them sequentially as they become available.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
            - "total_chargers" (int): The number of available chargers.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "charging_queue" (list[int]): The queue of EVs waiting to charge operator that schedules charging actions for each EV.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]
    charging_queue = state_data["charging_queue"]
    
    actions = [0] * fleet_size  # Initialize all actions to 0 (no charging)
    
    available_chargers = total_chargers - sum(charging_lead_time)
    
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    # Assign chargers to EVs based on the order they are in the queue
    for i in charging_queue:
        if available_chargers > 0 and ride_lead_time[i] == 0:
            actions[i] = 1
            available_chargers -= 1
        elif available_chargers <= 0:
            break

    return ActionOperator(actions), {}