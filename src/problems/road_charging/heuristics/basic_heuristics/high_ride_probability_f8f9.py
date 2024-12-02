from src.problems.base.mdp_components import Solution, ActionOperator

def high_ride_probability_f8f9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to align idle time with hours of high probability of receiving ride orders.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of EVs in the fleet.
            - max_time_steps (int): Maximum number of time steps.
            - assign_prob (list[float]): Probability of receiving a ride order at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - ride_lead_time (list[int]): Remaining ride time for each vehicle.
            - battery_soc (list[float]): State of charge of each vehicle.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): No specific data required for this heuristic.
        (Optional and can be omitted if no used) get_state_data_function (callable): Not used in this implementation.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.

    Returns:
        An ActionOperator that sets actions for vehicles to maximize ride opportunities.
        An empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from global_data and state_data
    fleet_size = global_data["fleet_size"]
    max_time_steps = global_data["max_time_steps"]
    assign_prob = global_data["assign_prob"]

    ride_lead_time = state_data["ride_lead_time"]

    # Initialize actions for each EV
    actions = [0] * fleet_size  # Start with all EVs idle

    # Iterate over each EV to set actions based on ride probability
    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            # If the vehicle is on a ride, it should not charge
            actions[i] = 0
        else:
            # If idle and can charge, set action based on ride probability
            actions[i] = 0  # Stay idle to maximize ride opportunities

    # Create an ActionOperator with the determined actions
    operator = ActionOperator(actions)

    return operator, {}