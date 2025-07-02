from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_34e6(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Updated Heuristic Strategy.

    Args:
        global_data (dict): Contains global data necessary for the algorithm.
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - max_time_steps (int): Maximum number of time steps.
        state_data (dict): Contains the current state information.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Algorithm-specific data if necessary, optional.
        get_state_data_function (callable): Function to receive the new solution and return the state dictionary for new solution.
        base_charge_lb (float, optional): Base lower bound of SoC for charging decisions, default is 0.78.
        base_charge_ub (float, optional): Base upper bound of SoC for staying available, default is 0.79.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.78)
    base_charge_ub = kwargs.get('base_charge_ub', 0.79)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    peak_customer_arrivals = max(customer_arrivals) if customer_arrivals else 0
    average_customer_arrivals = np.mean(customer_arrivals) if customer_arrivals else 0

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low time_to_next_availability and battery_soc
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= base_charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # If average customer arrivals significantly lower than peak, prioritize readiness
    if average_customer_arrivals < peak_customer_arrivals * 0.8:
        for i in range(fleet_size):
            if operational_status[i] == 0 and battery_soc[i] < base_charge_lb and sum(actions) < total_chargers:
                actions[i] = 1

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}