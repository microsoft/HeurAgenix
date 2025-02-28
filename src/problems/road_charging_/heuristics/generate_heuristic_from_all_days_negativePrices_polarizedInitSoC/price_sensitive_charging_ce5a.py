from src.problems.base.mdp_components import Solution, ActionOperator

def price_sensitive_charging_ce5a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to schedule EV charging during periods of lowest charging prices.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - total_chargers (int): Total number of chargers available.
            - charging_price (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): The index of the current time step.
            - ride_lead_time (list[int]): Ride leading time for each vehicle, where a value >= 2 indicates the vehicle is on a ride.
            - battery_soc (list[float]): State of charge of the battery for each vehicle, with 1 representing full charge.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not require any specific algorithm data.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.
        
    Returns:
        ActionOperator: An operator that modifies the Solution by scheduling charging actions based on price sensitivity and constraints.
        dict: The updated algorithm data (empty for this algorithm).
    """
    # Initialize an action list with zeros for all vehicles
    fleet_size = len(state_data['ride_lead_time'])
    actions = [0] * fleet_size
    
    # Retrieve necessary global and state data
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']
    current_step = state_data['current_step']
    ride_lead_time = state_data['ride_lead_time']
    battery_soc = state_data['battery_soc']
    
    # Create a list of eligible vehicles for charging
    eligible_vehicles = [
        i for i in range(fleet_size)
        if ride_lead_time[i] < 2 and battery_soc[i] < 1
    ]
    
    # Sort eligible vehicles by their battery state of charge (SoC), prioritize lower SoC
    eligible_vehicles.sort(key=lambda i: battery_soc[i])
    
    # Determine the number of vehicles we can charge based on available chargers and price
    chargers_available = min(total_chargers, len(eligible_vehicles))
    
    # Assign charging actions to vehicles with the lowest SoC, respecting the number of available chargers
    for i in range(chargers_available):
        actions[eligible_vehicles[i]] = 1
    
    # Return the operator and an empty dictionary for algorithm data
    return ActionOperator(actions), {}