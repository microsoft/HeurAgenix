from src.problems.base.mdp_components import Solution, ActionOperator

def balance_charging_and_availability_f036(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Alternate between charging and being available for rides to ensure EVs are not left with low battery and can take advantage of high payment ride opportunities.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - max_time_steps (int): Maximum number of time steps.
            - total_chargers (int): Total number of chargers.
            - order_price (list[float]): Payments received per time step when a vehicle is on a ride.
            - charging_price (list[float]): Charging price per kilowatt-hour at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - ride_lead_time (list[int]): Ride leading time for each vehicle in the fleet.
            - battery_soc (list[float]): State of charge for each vehicle in the fleet.
        (Optional) algorithm_data (dict): The algorithm dictionary for current algorithm only.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the original solution.
        (Optional) introduction for hyper parameters in kwargs if used:
            - low_battery_threshold (float): Default=0.1, the threshold of state of charge below which EVs should prioritize charging.

    Returns:
        ActionOperator: An operator to modify the solution with new actions for the current time step.
        dict: An empty dictionary as no updates to algorithm_data are necessary in this implementation.
    """
    # Hyperparameters
    low_battery_threshold = kwargs.get('low_battery_threshold', 0.1)
    
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    order_price = global_data['order_price']
    current_time_step = len(state_data['ride_lead_time'])
    
    # Determine current time step based on ride_lead_time length
    if current_time_step >= global_data['max_time_steps']:
        # Return an empty operator if all time steps have been processed
        return ActionOperator([0] * fleet_size), {}

    # Calculate actions based on current state
    actions = []
    for i in range(fleet_size):
        # If the EV is on a ride, it cannot charge
        if state_data['ride_lead_time'][i] > 0:
            actions.append(0)
        else:
            # Determine if EV should charge
            if state_data['battery_soc'][i] < low_battery_threshold or (order_price[current_time_step] < global_data['charging_price'][current_time_step]):
                actions.append(1)
            else:
                actions.append(0)

    # Return the operator with the new actions
    return ActionOperator(actions), {}