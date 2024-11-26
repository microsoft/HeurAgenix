from src.problems.base.mdp_components import Solution, ActionOperator

def profit_maximization_4f39(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Prioritize charging for EVs that have the potential to earn the most from subsequent rides, based on their location and expected demand.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Total number of EVs in the fleet.
            - "total_chargers" (int): Maximum number of chargers available.
            - "order_price" (list[float]): List of order prices for each time step.
            - "assign_prob" (float): Probability of receiving a ride order when idle.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[int]): State of charge (SoC) for each EV.
            - "ride_lead_time" (list[int]): Remaining ride time for each EV.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        kwargs: No additional hyper-parameters are used in this implementation.

    Returns:
        An ActionOperator that specifies the charging decisions for each EV based on maximizing potential earnings.
        An empty dictionary as no algorithm data needs to be updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    order_price = global_data["order_price"]
    assign_prob = global_data["assign_prob"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Initialize actions with zero (no charging)
    actions = [0] * fleet_size

    # Calculate potential earnings for each EV if they receive a ride
    potential_earnings = [
        (order_price[0] * battery_soc[i] * assign_prob) if ride_lead_time[i] == 0 else 0
        for i in range(fleet_size)
    ]

    # Sort EVs based on potential earnings in descending order
    sorted_ev_indices = sorted(range(fleet_size), key=lambda i: potential_earnings[i], reverse=True)

    # Allocate chargers to EVs with highest potential earnings
    chargers_used = 0
    for i in sorted_ev_indices:
        if chargers_used < total_chargers and ride_lead_time[i] == 0:
            actions[i] = 1
            chargers_used += 1

    # Create and return the operator with the actions decided
    operator = ActionOperator(actions)
    return operator, {}
