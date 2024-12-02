from src.problems.base.mdp_components import Solution, ActionOperator

def prioritize_low_soc_4f7e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Prioritize charging for EVs with a low state of charge (SoC) to ensure they are ready for future ride assignments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): SoC of battery of each fleet, length is fleet_size.
            - "ride_lead_time" (list[int]): Ride leading time, length is fleet_size.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.
            - "threshold" (float, default=0.2): The SoC threshold below which an EV should be prioritized for charging.

    Returns:
        An ActionOperator that defines the charging actions for each EV in the fleet.
        An empty dictionary as no algorithm data is updated in this heuristic.
    """
    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]
    
    # Hyper-parameter for SoC threshold
    threshold = kwargs.get("threshold", 0.2)
    
    # Initialize actions to not charge all vehicles
    actions = [0] * fleet_size
    
    # Determine actions based on SoC and ride status
    for i in range(fleet_size):
        if ride_lead_time[i] == 0 and battery_soc[i] < threshold:
            # If the vehicle is not on a ride and SoC is below threshold, schedule it for charging
            actions[i] = 1
    
    # Create an ActionOperator with the determined actions
    operator = ActionOperator(actions)
    
    # Return the operator and an empty dictionary as no algorithm data is updated
    return operator, {}