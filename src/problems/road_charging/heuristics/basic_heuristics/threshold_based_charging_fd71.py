from src.problems.base.mdp_components import Solution, ActionOperator

def threshold_based_charging_fd71(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implements the ThresholdBasedCharging heuristic.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of EVs in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): The state of charge (SoC) of each EV. Length is fleet_size.
            - "ride_lead_time" (list[int]): The remaining ride time for each EV. Length is fleet_size.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        kwargs: 
            - "threshold" (float, default=0.2): The SoC threshold below which an EV should start charging.

    Returns:
        An ActionOperator with the charging actions for each EV based on the SoC threshold.
        An empty dictionary, as this algorithm does not update algorithm_data.
    """
    # Set default threshold for charging
    threshold = kwargs.get("threshold", 0.2)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Initialize actions
    actions = [0] * fleet_size

    # Determine actions based on SoC threshold and ride status
    for i in range(fleet_size):
        if ride_lead_time[i] == 0 and battery_soc[i] < threshold:
            actions[i] = 1  # Initiate charging if not on a ride and SoC is below threshold

    # Create the ActionOperator with the determined actions
    operator = ActionOperator(actions)

    # Return the operator and an empty dictionary (no updates to algorithm_data)
    return operator, {}