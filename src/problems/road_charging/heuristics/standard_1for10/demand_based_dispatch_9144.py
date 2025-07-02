from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_based_dispatch_9144(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ DemandBasedDispatch Heuristic: Allocates EVs to remain available or go to charge based on dynamic thresholds and historical performance data.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Total number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total available charging stations.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "current_step" (int): Current time step index.
            - "operational_status" (np.ndarray): Status of each EV (0: idle, 1: serving, 2: charging).
            - "time_to_next_availability" (np.ndarray): Time until each EV is available.
            - "battery_soc" (np.ndarray): Battery state of charge for each EV.

        get_state_data_function (callable): Function to get state data for a potential new solution. Not used in this basic implementation.

        **kwargs: Optional hyper-parameters for this heuristic:
            - "base_low_soc_threshold" (float, default=0.3): Base threshold below which EVs are prioritized for charging.
            - "high_demand_threshold" (int, default=5): Threshold of customer arrivals to consider it a high demand period.
            - "rolling_window_size" (int, default=3): Window size for calculating the rolling average of demand.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Dictionary containing updated algorithm data based on fleet performance metrics.
    """
    
    base_low_soc_threshold = kwargs.get('base_low_soc_threshold', 0.3)
    high_demand_threshold = kwargs.get('high_demand_threshold', 5)
    rolling_window_size = kwargs.get('rolling_window_size', 3)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Calculate rolling average demand
    rolling_demand = np.mean(customer_arrivals[max(0, current_step - rolling_window_size):current_step + 1])

    # Dynamically adjust low SoC threshold based on rolling average demand
    low_soc_threshold = base_low_soc_threshold
    if rolling_demand > high_demand_threshold:
        low_soc_threshold -= 0.05  # Slightly lower threshold during high rolling average demand

    # Determine the number of EVs that can charge based on charger availability
    available_chargers = total_chargers

    # Apply dynamic charging prioritization logic
    missed_demand_opportunities = 0
    overcharging_incidents = 0
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:  # EV is on a ride
            actions[i] = 0  # Must remain available
        elif battery_soc[i] < low_soc_threshold and available_chargers > 0:
            actions[i] = 1  # Set to charge
            available_chargers -= 1
        else:
            missed_demand_opportunities += 1

    # Update algorithm data with fleet performance metrics
    algorithm_data_updates = {
        "missed_demand_opportunities": missed_demand_opportunities,
        "overcharging_incidents": overcharging_incidents
    }

    # Ensure the sum of actions does not exceed total chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    return ActionOperator(actions), algorithm_data_updates