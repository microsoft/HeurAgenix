from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_based_dispatch_5b67(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ DemandBasedDispatch Heuristic with Adaptive Thresholds and Dynamic Rolling Window: Allocates EVs to remain available or go to charge based on real-time demand patterns.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Total number of EVs in the fleet.
            - max_time_steps (int): Maximum number of time steps.
            - total_chargers (int): Total available charging stations.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - max_SoC (float): Maximum allowable state of charge for the EV batteries.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory for EVs.
            - current_step (int): Current time step index.
            - operational_status (np.ndarray): Status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (np.ndarray): Time until each EV is available.
            - battery_soc (np.ndarray): Battery state of charge for each EV.
        get_state_data_function (callable): Not used in this implementation.
        kwargs: Optional hyper-parameters for this heuristic:
            - base_low_soc_threshold (float, default=0.35): Base threshold for charging.
            - initial_rolling_window_size (int, default=3): Initial window size for calculating rolling demand average.
            - demand_volatility_threshold (float, default=0.1): Threshold for adjusting SoC thresholds based on demand volatility.
            - min_window_size (int, default=1): Minimum rolling window size.
            - max_window_size (int, default=5): Maximum rolling window size.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Dictionary containing updated algorithm data based on fleet performance metrics.
    """
    
    # Unpacking hyper-parameters with default values
    base_low_soc_threshold = kwargs.get('base_low_soc_threshold', 0.35)
    initial_rolling_window_size = kwargs.get('initial_rolling_window_size', 3)
    demand_volatility_threshold = kwargs.get('demand_volatility_threshold', 0.1)
    min_window_size = kwargs.get('min_window_size', 1)
    max_window_size = kwargs.get('max_window_size', 5)

    # Extract necessary global and state data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Calculate demand volatility
    demand_volatility = np.std(customer_arrivals[max(0, current_step - initial_rolling_window_size):current_step + 1])

    # Dynamically adjust rolling window size based on demand volatility
    if demand_volatility > demand_volatility_threshold:
        rolling_window_size = max(min_window_size, initial_rolling_window_size - 1)
    else:
        rolling_window_size = min(max_window_size, initial_rolling_window_size + 1)

    # Calculate rolling average demand
    rolling_demand = np.mean(customer_arrivals[max(0, current_step - rolling_window_size):current_step + 1])

    # Adjust thresholds based on demand volatility
    if demand_volatility > demand_volatility_threshold:
        adaptive_low_soc_threshold = base_low_soc_threshold * 0.9  # decrease threshold to charge earlier
    else:
        adaptive_low_soc_threshold = base_low_soc_threshold

    mid_soc_threshold = (adaptive_low_soc_threshold + global_data["max_SoC"]) / 2

    # Determine the number of EVs that can charge based on charger availability
    available_chargers = total_chargers

    # Apply logic to prioritize charging
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:  # EV is on a ride
            actions[i] = 0  # Must remain available
        elif battery_soc[i] < adaptive_low_soc_threshold and available_chargers > 0:
            actions[i] = 1  # Prioritize charging for low SoC
            available_chargers -= 1
        elif battery_soc[i] < mid_soc_threshold and available_chargers > 0:
            # Check for significant demand increase
            if (customer_arrivals[current_step] - rolling_demand) > demand_volatility_threshold:
                actions[i] = 1  # Prioritize charging for mid SoC during high demand
                available_chargers -= 1

    # Ensure the sum of actions does not exceed total chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    # Return operator and update algorithm data
    return ActionOperator(actions), {}