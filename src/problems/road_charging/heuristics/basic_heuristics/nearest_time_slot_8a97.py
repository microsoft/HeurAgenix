from src.problems.base.mdp_components import *
import numpy as np

def nearest_time_slot_8a97(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Nearest Time Slot heuristic for road_charging.

    Args:
        global_data (dict): The global data dict containing the global instance data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "charging_rate" (list[float]): Battery charging rate per time step for each vehicle.
            - "charging_price" (list[float]): Charging price per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the solution state data. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time for each EV.
            - "battery_soc" (list[float]): State of charge (SoC) of each EV.
            - "charging_lead_time" (list[int]): Charging lead time for each EV.
        kwargs: 
            - "priority_soc_threshold" (float): Optional threshold to prioritize EVs with SoC below this value. Default is 0.2 (20%).

    Returns:
        ActionOperator: The operator to schedule charging actions for EVs.
        dict: Empty dictionary as this algorithm does not update the algorithm data.
    """
    fleet_size = global_data['fleet_size']
    max_time_steps = global_data['max_time_steps']
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']
    
    ride_lead_time = state_data['ride_lead_time']
    battery_soc = state_data['battery_soc']
    
    priority_soc_threshold = kwargs.get('priority_soc_threshold', 0.2)

    # Initialize actions for all EVs as not charging
    actions = [0] * fleet_size

    # Sort EVs by SoC, prioritizing those with SoC below the threshold
    ev_priority = sorted(range(fleet_size), key=lambda i: (battery_soc[i] > priority_soc_threshold, battery_soc[i]))

    # Attempt to schedule charging actions
    chargers_in_use = sum(state_data["charging_lead_time"])
    for ev in ev_priority:
        if ride_lead_time[ev] > 0:
            # Skip EVs that are currently on a ride
            continue
        
        if chargers_in_use >= total_chargers:
            # No more chargers available
            break

        # Find the nearest time slot with the lowest price
        min_price_time = np.argmin(charging_price)
        if charging_price[min_price_time] < np.min(charging_price) * 1.1:  # 10% above minimum price
            actions[ev] = 1
            chargers_in_use += 1
    
    # Return ActionOperator with the computed actions
    return ActionOperator(actions), {}