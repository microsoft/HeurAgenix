from src.problems.base.mdp_components import Solution, ActionOperator

def maximize_ride_payment_7f47(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Align EV availability with hours of maximum ride payment rates, ensuring that vehicles are available to take advantage of higher earnings per ride time.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "order_price" (list[float]): Ride order payment per step at each hour.
            - "max_time_steps" (int): Maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Current ride lead time for each EV.
            - "battery_soc" (list[float]): Battery state of charge for each EV.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this algorithm.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, without modifying the original solution.

    Returns:
        ActionOperator: An operator with actions keeping EVs available during high payment hours if feasible.
        dict: An empty dictionary as algorithm data is not updated in this heuristic.
    """
    # Extract required data
    order_price = global_data["order_price"]
    max_time_steps = global_data["max_time_steps"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Determine the top payment hours (e.g., top 10%)
    top_payment_threshold = 0.1  # This can be a hyper-parameter
    top_payment_hours = sorted(range(len(order_price)), key=lambda i: order_price[i], reverse=True)[:int(max_time_steps * top_payment_threshold)]

    # Initialize actions
    actions = [0] * len(ride_lead_time)  # Default to not charging

    # Iterate over each EV
    for i in range(len(ride_lead_time)):
        if ride_lead_time[i] == 0:  # EV is idle
            # If current hour is a top payment hour, keep the EV available (do not charge)
            current_hour = i % 24  # Assuming each time step corresponds to an hour in a day
            if current_hour in top_payment_hours:
                actions[i] = 0  # Stay idle to take rides
            else:
                # Default action could be to charge if needed (not implemented here)
                pass
        else:
            actions[i] = 0  # On ride, cannot charge

    # Create ActionOperator with the determined actions
    operator = ActionOperator(actions)
    return operator, {}