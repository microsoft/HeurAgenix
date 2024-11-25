from src.problems.base.mdp_components import *
import numpy as np

def charging_saving_strategy_0a5f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Charging Saving Strategy heuristic for the road_charging problem.
    
    This algorithm calculates potential savings for scheduling charging sessions at different times for each EV in the fleet. It evaluates the benefit of shifting charging times to periods of lower cost or higher renewable energy availability, aiming to reduce overall charging costs while respecting the charging station capacity.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
            - "total_chargers" (int): The number of available chargers.
            - "charging_price" (list[float]): The price per kWh at each time step.
            - "renewable_energy_availability" (list[float]): The availability of renewable energy at each time step.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[int]): The state of charge for each EV.
            - "ride_lead_time" (list[int]): The remaining ride time if an EV is currently on a ride.
            - "charging_lead_time" (list[int]): The time an EV has been charging.

    Returns:
        ActionOperator: The operator that schedules charging actions for each EV based on calculated savings.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    renewable_energy_availability = global_data["renewable_energy_availability"]
    
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]
    
    actions = [0] * fleet_size  # Initialize all actions to 0 (no charging)
    
    available_chargers = total_chargers - sum(charging_lead_time)
    
    if available_chargers <= 0:
        return ActionOperator(actions), {}

    savings = []

    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            continue  # Do not charge if EV is on a ride

        # Calculate the potential savings of charging at this time slot
        current_cost = charging_price[i] * battery_soc[i]
        potential_savings = current_cost * (1 - renewable_energy_availability[i])
        savings.append((potential_savings, i))
    
    # Sort by highest savings
    savings.sort(reverse=True, key=lambda x: x[0])
    
    for s, i in savings:
        if available_chargers > 0:
            actions[i] = 1
            available_chargers -= 1
        else:
            break

    return ActionOperator(actions), {}