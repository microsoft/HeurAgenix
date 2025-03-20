from src.problems.base.mdp_components import *
# Assuming necessary imports for Solution and ActionOperator are already done.

def greedy_charging_c778(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Greedy Charging Heuristic Algorithm.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - total_chargers (int): Total number of chargers available.
            - charging_price (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): The index of current time step.
            - ride_lead_time (list[int]): Ride leading time for each vehicle. 
            - battery_soc (list[float]): State of charge of each vehicle's battery.
        (Optional) introduction for hyper parameters in kwargs if used.

    Returns:
        ActionOperator: The operator that applies the charging actions to the solution.
        dict: Updated algorithm data, empty in this case.
    """
    current_step = state_data['current_step']
    ride_lead_time = state_data['ride_lead_time']
    battery_soc = state_data['battery_soc']
    charging_price = global_data['charging_price'][current_step]
    total_chargers = global_data['total_chargers']
    fleet_size = global_data['fleet_size']

    # Initialize actions for the fleet
    actions = [0] * fleet_size
    
    # Gather indices of vehicles that can charge (not fully charged, not on ride)
    chargeable_vehicles = [
        i for i in range(fleet_size) 
        if ride_lead_time[i] < 2 and battery_soc[i] < 1
    ]
    
    # Sort vehicles by state of charge (SoC) in ascending order to prioritize lower SoC
    chargeable_vehicles.sort(key=lambda i: battery_soc[i])
    
    # Apply actions to charge vehicles with the lowest SoC first
    chargers_used = 0
    for i in chargeable_vehicles:
        if chargers_used < total_chargers:
            actions[i] = 1  # Schedule for charging
            chargers_used += 1
        else:
            break  # Stop if all chargers are used

    # Return ActionOperator with the actions determined
    return ActionOperator(actions), {}