from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_c920(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Context-Sensitive Penalty and Reward Factors.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Not used in this implementation.
        get_state_data_function (callable): Not used in this implementation.
        kwargs (dict): Hyper-parameters for fine-tuning the algorithm. Includes:
            - base_charge_lb (float, optional): Base lower bound of SoC for charging, default is 0.35.
            - base_charge_ub (float, optional): Base upper bound of SoC for staying available, default is 0.45.
            - soc_threshold (float, optional): Threshold for applying penalty factor, default is 0.6.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, complying with constraints.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.35)
    base_charge_ub = kwargs.get('base_charge_ub', 0.45)
    soc_threshold = kwargs.get('soc_threshold', 0.6)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']

    # Calculate dynamic thresholds using a weighted moving average
    soc_window = battery_soc[max(0, current_step-5):current_step+1]
    avg_soc = np.average(soc_window, weights=np.linspace(0.5, 1, len(soc_window)))
    demand_window = global_data['customer_arrivals'][max(0, current_step-5):current_step+1]
    demand_fluctuation = np.std(demand_window)

    # Apply penalty factor only if the average SoC is above the threshold
    penalty_factor = 0.1 * (sum(operational_status) / fleet_size) if avg_soc > soc_threshold else 0

    # Introduce a reward factor for maintaining high availability rates
    reward_factor = 0.05 * (fleet_size - sum(operational_status))

    # Adjust charge_lb and charge_ub with context-sensitive penalty and reward factors
    charge_lb = base_charge_lb + (0.1 * (0.5 - avg_soc)) - (0.05 * demand_fluctuation) - penalty_factor + reward_factor
    charge_ub = base_charge_ub + (0.1 * (avg_soc - 0.5)) + (0.05 * demand_fluctuation) + penalty_factor - reward_factor

    # Ensure thresholds are within valid bounds
    charge_lb = max(0.1, min(charge_lb, 0.9))
    charge_ub = max(0.1, min(charge_ub, 0.9))

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC and idle for more than one step
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb and operational_status[i] == 0:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Sort EVs by battery SoC to prioritize those with the lowest charge
    sorted_indices = sorted(range(fleet_size), key=lambda x: battery_soc[x])
    for i in sorted_indices:
        if sum(actions) > total_chargers:
            actions[i] = 0

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