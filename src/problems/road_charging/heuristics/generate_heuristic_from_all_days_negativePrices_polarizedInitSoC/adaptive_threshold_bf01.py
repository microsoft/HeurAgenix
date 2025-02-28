from src.problems.base.mdp_components import Solution, ActionOperator

def adaptive_threshold_bf01(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implement an adaptive SoC threshold heuristic for scheduling EV charging.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
            - "charging_price" (list[float]): Charging price in dollars per kWh at each time step.
            - "order_price" (list[float]): Payments received per time step when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Remaining time on ride for each vehicle.
            - "battery_soc" (list[float]): State of charge for each vehicle.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no specific items are necessary.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        introduction for hyper parameters in kwargs if used:
            - "sensitivity" (float): The sensitivity parameter for adaptive threshold calculation. Default value is 0.5.

    Returns:
        An ActionOperator that modifies the current solution by scheduling EVs for charging based on the adaptive threshold logic.
        An empty dictionary as no algorithm-specific data is updated.
    """
    sensitivity = kwargs.get("sensitivity", 0.5)  # Default sensitivity value for threshold calculation

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    order_price = global_data["order_price"]
    
    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Calculate adaptive thresholds and determine charging actions
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2:
            # Vehicle is on a ride, cannot charge
            actions[i] = 0
            continue
        if battery_soc[i] >= 1:
            # Vehicle is fully charged, do not charge
            actions[i] = 0
            continue
        
        # Calculate the adaptive threshold
        lambda_i = battery_soc[i] + sensitivity * (order_price[current_step] - charging_price[current_step])
        
        # Determine if the vehicle should start charging
        if battery_soc[i] < lambda_i:
            actions[i] = 1

    # Ensure the sum of actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        # Sort vehicles by priority to determine which should charge
        priorities = [(battery_soc[i], i) for i in range(fleet_size) if actions[i] == 1]
        priorities.sort()  # Sort by SoC, ascending
        
        # Set the lowest priority vehicles to not charge
        for _, i in priorities[total_chargers:]:
            actions[i] = 0

    # Create and return the ActionOperator
    return ActionOperator(actions), {}