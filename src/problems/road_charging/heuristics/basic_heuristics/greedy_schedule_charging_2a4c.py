from src.problems.base.mdp_components import ActionOperator

def greedy_schedule_charging_2a4c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Greedy Schedule Charging heuristic for the road_charging problem.

    This heuristic aims to minimize charging costs by scheduling EVs to charge during the lowest-cost available time slots, while respecting charger availability and battery SoC constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
            - "charging_price" (list[float]): Charging price in dollars per kWh at each time step.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride lead time for each vehicle.
            - "battery_soc" (list[float]): State of charge (SoC) for each vehicle in the fleet.
        
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution. It will not modify the original solution.

    Returns:
        ActionOperator: An operator scheduling charging actions for the EVs at the current time step.
        dict: An empty dictionary as this algorithm does not update algorithm-specific data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    
    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # List of EV indices sorted by the greatest need for charging (lowest SoC)
    ev_indices = sorted(range(fleet_size), key=lambda i: battery_soc[i])
    
    # Track the number of chargers currently being used
    chargers_used = 0
    
    for i in ev_indices:
        # Check if the EV is on a ride or fully charged
        if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
            continue
        
        # Check if we can schedule more charging without exceeding total chargers
        if chargers_used < total_chargers:
            actions[i] = 1
            chargers_used += 1
        else:
            break
    
    return ActionOperator(actions), {}