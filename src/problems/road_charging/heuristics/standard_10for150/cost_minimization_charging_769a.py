from src.problems.base.mdp_components import *
import numpy as np
from collections import deque

def cost_minimization_charging_769a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Demand Spike Reset Mechanism.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        Hyper-parameters:
            - soc_proximity_threshold (float, optional): SoC proximity threshold to consider charging, default is 0.02.
            - demand_factor (float, optional): Factor to adjust charging threshold based on customer demand, default is 0.1.
            - rolling_window_size (int, optional): Size of the rolling window for average calculation, default is 5.
            - demand_spike_threshold (float, optional): Threshold for detecting a demand spike, default is 0.2.
            - reset_window_size (int, optional): Size of the window for resetting adaptive charge threshold, default is 3.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set default values for hyper-parameters
    soc_proximity_threshold = kwargs.get('soc_proximity_threshold', 0.02)
    demand_factor = kwargs.get('demand_factor', 0.1)
    rolling_window_size = kwargs.get('rolling_window_size', 5)
    demand_spike_threshold = kwargs.get('demand_spike_threshold', 0.2)
    reset_window_size = kwargs.get('reset_window_size', 3)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']

    # Initialize rolling window deques for customer arrivals and battery SoC
    past_customer_arrivals = deque(maxlen=rolling_window_size)
    past_battery_soc = deque(maxlen=rolling_window_size)

    # Populate rolling windows with past data if available
    for t in range(max(0, current_step - rolling_window_size), current_step):
        past_customer_arrivals.append(customer_arrivals[t])
        past_battery_soc.append(np.mean(battery_soc))

    # Calculate rolling window averages
    rolling_avg_demand = np.mean(past_customer_arrivals) if past_customer_arrivals else customer_arrivals[current_step]
    rolling_avg_soc = np.mean(past_battery_soc) if past_battery_soc else np.mean(battery_soc)

    # Calculate adaptive charge_lb based on rolling window averages
    peak_demand = max(customer_arrivals)
    adaptive_charge_lb = 0.65 + demand_factor * (rolling_avg_demand / peak_demand)

    # Detect demand spike
    current_demand = customer_arrivals[current_step]
    if past_customer_arrivals and current_demand > rolling_avg_demand * (1 + demand_spike_threshold):
        adaptive_charge_lb -= 0.05  # Temporarily lower threshold during demand spike

    # Check if demand has subsided and reset the adaptive charge threshold
    reset_window = deque(maxlen=reset_window_size)
    for t in range(max(0, current_step - reset_window_size), current_step):
        reset_window.append(customer_arrivals[t])
    if np.mean(reset_window) <= rolling_avg_demand:
        adaptive_charge_lb = 0.65 + demand_factor * (rolling_avg_demand / peak_demand)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery_soc
        elif time_to_next_availability[i] == 0 and (battery_soc[i] <= adaptive_charge_lb or abs(battery_soc[i] - rolling_avg_soc) <= soc_proximity_threshold):
            actions[i] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}