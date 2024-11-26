from src.problems.base.mdp_components import Solution, ActionOperator

def maximize_idle_time_charging_58b9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Focus on charging EVs when they are expected to be idle for longer durations to maximize charging efficiency.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Estimated idle time for each EV in the fleet.
            - "battery_soc" (list[float]): Current state of charge for each EV in the fleet.
            - "reward" (float): The total reward for all fleets for this time step.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        An ActionOperator that sets charging actions for selected EVs based on their expected idle time.
        An empty dictionary as no updates are made to the algorithm data.
    """
    # Extract necessary data from input dictionaries
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions for all EVs as not charging
    actions = [0] * len(ride_lead_time)

    # Create a list of (idle_time, EV_index) and sort by idle time in descending order
    idle_time_with_index = [(ride_lead_time[i], i) for i in range(len(ride_lead_time))]
    idle_time_with_index.sort(reverse=True, key=lambda x: x[0])

    # Select EVs for charging based on available chargers
    chargers_used = 0
    for idle_time, i in idle_time_with_index:
        # Check if EV is idle and not on a ride
        if idle_time > 0 and battery_soc[i] < 1.0:
            actions[i] = 1  # Schedule to charge
            chargers_used += 1
            if chargers_used >= total_chargers:
                break

    # Create and return ActionOperator with the determined actions
    operator = ActionOperator(actions)
    return operator, {}