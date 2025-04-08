from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_0ef9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Granular Dynamic Charging Optimization for EV Fleet.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - max_time_steps (int): Total number of time steps in the simulation.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no specific data is necessary.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the original solution.
        Introduction for hyper parameters in kwargs:
            - base_charge_lb (float): Base lower bound of battery SoC for charging decision, default is 0.65.
            - base_charge_ub (float): Base upper bound of battery SoC for charging decision, default is 0.75.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    # Set default values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.65)
    base_charge_ub = kwargs.get('base_charge_ub', 0.75)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    max_time_steps = global_data['max_time_steps']  # Corrected extraction of max_time_steps
    average_customer_arrivals = np.mean(customer_arrivals) if customer_arrivals else 0

    # Determine historical peak demand times
    historical_peak_times = [i for i, arrival in enumerate(customer_arrivals) if arrival == max(customer_arrivals)]
    upcoming_peak = min(historical_peak_times, key=lambda x: abs(x - current_step)) if historical_peak_times else current_step

    # Weighted decision factor based on immediate SoC and future demand prediction
    weighted_factors = [(battery_soc[i] * 0.7) + ((upcoming_peak - current_step) / max_time_steps * 0.3) for i in range(fleet_size)]

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Limit the application scope of the charging logic to scenarios with high fleet-to-charger ratio
    if fleet_size / total_chargers > 5:
        # Determine actions for each EV
        for i in range(fleet_size):
            # Ensure EVs on a ride remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # Prioritize charging for idle EVs with low battery SoC
            elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
                actions[i] = 1
            elif time_to_next_availability[i] == 0 and battery_soc[i] >= base_charge_ub:
                actions[i] = 0

        # Adjust actions to prioritize the EV with the lowest weighted factor for charging
        idle_indices = [i for i in range(fleet_size) if time_to_next_availability[i] == 0]
        if idle_indices:
            min_factor_index = min(idle_indices, key=lambda idx: weighted_factors[idx])
            actions = [0] * fleet_size
            actions[min_factor_index] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}