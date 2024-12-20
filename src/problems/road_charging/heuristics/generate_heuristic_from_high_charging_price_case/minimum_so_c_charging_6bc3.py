from src.problems.base.mdp_components import Solution, ActionOperator

def minimum_so_c_charging_6bc3(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to prioritize charging EVs with a state of charge (SoC) below a threshold.

    Args:
        global_data (dict): The global data dict containing the global instance data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): List of SoC for each vehicle in the fleet.
            - "ride_lead_time" (list[int]): Remaining time steps for each vehicle currently on a ride.
        (Optional) algorithm_data (dict): Not used in this heuristic.
        (Optional) get_state_data_function (callable): Not used in this heuristic.
        kwargs: 
            - "low_battery_threshold" (float, default=0.1): The SoC threshold below which vehicles should be prioritized for charging.

    Returns:
        An ActionOperator that modifies the current solution to prioritize charging for low SoC vehicles.
        An empty dictionary, as no algorithm-specific data needs to be updated.
    """
    # Extract necessary parameters from global_data and state_data
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    
    # Extract hyper-parameters from kwargs
    low_battery_threshold = kwargs.get("low_battery_threshold", 0.1)
    
    # Initialize actions with zeros (no charging by default)
    actions = [0] * len(battery_soc)
    
    # Count of available chargers
    chargers_used = 0

    # Iterate over each vehicle to determine charging actions
    for i in range(len(battery_soc)):
        # Check constraints: vehicle not on a ride, SoC below threshold, and available chargers
        if ride_lead_time[i] < 2 and battery_soc[i] < low_battery_threshold and chargers_used < total_chargers:
            actions[i] = 1  # Schedule this vehicle for charging
            chargers_used += 1  # Increment the count of chargers used

    # Create an ActionOperator with the determined actions
    operator = ActionOperator(actions)
    
    return operator, {}