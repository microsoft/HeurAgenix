from src.problems.base.mdp_components import *
import numpy as np

def nearest_benefit_charging_5500(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Nearest Benefit Charging heuristic for the road_charging problem.
    
    This algorithm iterates over each EV to determine the charging action that provides the highest immediate benefit, considering charging station capacity. It evaluates potential benefits based on cost savings and state of charge improvement, considering charging costs and potential ride earnings. The process continues until no further beneficial actions can be taken without exceeding capacity.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
            - "total_chargers" (int): The number of available chargers.
            - "charging_price" (list[float]): The price per kWh at each time step.
            - "order_price" (list[float]): The earnings per time step for each EV.
            - "assign_prob" (float): The probability of receiving a ride order when the vehicle is idle.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[int]): The state of charge for each EV.
            - "ride_lead_time" (list[int]): The remaining ride time if an EV is currently on a ride.
            - "charging_lead_time" (list[int]): The time an EV has been charging.

    Returns:
        ActionOperator: The operator that schedules charging actions for each EV.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    order_price = global_data["order_price"]
    assign_prob = global_data["assign_prob"]
    
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]
    
    actions = [0] * fleet_size  # Initialize all actions to 0 (no charging)
    
    available_chargers = total_chargers - sum(charging_lead_time)
    
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    benefit = []

    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            continue  # Do not charge if EV is on a ride

        # Calculate the potential benefit of charging
        potential_benefit = (order_price[i] * assign_prob) - (charging_price[i] * battery_soc[i])
        benefit.append((potential_benefit, i))
    
    # Sort by highest benefit
    benefit.sort(reverse=True, key=lambda x: x[0])
    
    for b, i in benefit:
        if available_chargers > 0:
            actions[i] = 1
            available_chargers -= 1
        else:
            break

    return ActionOperator(actions), {}