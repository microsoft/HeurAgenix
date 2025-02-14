from src.problems.base.mdp_components import *
import numpy as np

def charge_shift_between_schedules_9ab2(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Charge Shift Between Schedules heuristic for road_charging.
    
    This heuristic attempts to move a charging action from one time slot to another, aiming to reduce overall costs or improve load balance on the power grid.
    It evaluates the benefit of shifting an EV's charging action between different time slots or chargers, considering constraints like charger availability and battery state of charge.
    
    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
            
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "battery_soc" (list[float]): State of charge (SoC) of the battery for each vehicle.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
        
    Returns:
        ActionOperator: An operator scheduling the charging actions for the current time step.
        dict: An empty dictionary as no algorithm-specific data is updated.
    """
    # Extract data from global_data and state_data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Determine potential cost savings for each vehicle by shifting charging to current time step
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
            # If vehicle is on a ride or fully charged, it cannot be scheduled to charge
            continue
        
        # Calculate potential cost of charging at the current time step
        cost = charging_price[current_step]
        
        # If the cost is lower than a threshold, consider charging
        if cost < np.percentile(charging_price, 25):  # Example threshold: lower quartile
            actions[i] = 1
    
    # Ensure the total number of chargers is not exceeded
    if sum(actions) > total_chargers:
        # Select vehicles with the highest potential cost savings
        cost_savings = [(i, charging_price[current_step]) for i in range(fleet_size) if actions[i] == 1]
        cost_savings.sort(key=lambda x: x[1])
        for i, _ in cost_savings:
            if sum(actions) <= total_chargers:
                break
            actions[i] = 0
    
    # Return the ActionOperator and an empty dictionary
    return ActionOperator(actions), {}