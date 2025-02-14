from src.problems.base.mdp_components import Solution, ActionOperator

def time_based_charging_abbd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implements the TimeBasedCharging heuristic algorithm.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle, representing the remaining steps of a ride.
            - "battery_soc" (list[float]): State of charge of the battery for each vehicle.
        get_state_data_function (callable): The function that receives a new solution as input and returns the state dictionary for the new solution, without modifying the original solution.
        kwargs: Additional hyper-parameters for the algorithm.
            - "price_threshold" (float, default=0.1): The charging price threshold below which vehicles will be scheduled to charge.

    Returns:
        An ActionOperator that schedules charging for vehicles that meet the criteria.
        An empty dictionary for algorithm data updates.
    """
    # Extract necessary data
    charging_prices = global_data["charging_price"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]
    
    # Hyper-parameter
    price_threshold = kwargs.get("price_threshold", 0.1)  # Default charging price threshold

    # Initialize actions
    actions = [0] * len(battery_soc)  # Default to no charging

    # Check if the current charging price is below the threshold
    if charging_prices[current_step] < price_threshold:
        # Count the number of chargers in use
        chargers_in_use = 0
        for i in range(len(battery_soc)):
            # Skip vehicles on a ride or fully charged
            if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
                continue

            # Schedule charging if there are chargers available
            if chargers_in_use < total_chargers:
                actions[i] = 1
                chargers_in_use += 1

    # Ensure the total number of scheduled charges does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * len(battery_soc)  # Reset to no charging if exceeded

    # Create the operator
    operator = ActionOperator(actions)

    return operator, {}