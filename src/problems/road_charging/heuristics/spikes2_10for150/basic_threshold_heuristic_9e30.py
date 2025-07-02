from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_9e30(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Threshold Heuristic for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used directly, placeholder for function interface.
        introduction for hyper parameters in kwargs if used:
            - base_charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.76.
            - base_charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.78.
            - demand_window (int, optional): Number of time steps to consider for demand prediction. Default is 5.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Set default values for hyper-parameters
    base_charge_lb = kwargs.get('base_charge_lb', 0.76)
    base_charge_ub = kwargs.get('base_charge_ub', 0.78)
    demand_window = kwargs.get('demand_window', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']

    # Calculate future demand
    future_demand = np.mean(customer_arrivals[current_step:current_step + demand_window]) if current_step + demand_window < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Sort EVs by battery SoC to prioritize charging for EVs with lowest SoC
    sorted_indices = sorted(range(fleet_size), key=lambda i: battery_soc[i])

    # Determine actions for each EV
    for i in sorted_indices:
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= base_charge_ub:
            actions[i] = 0

    # Limit the sum of actions to the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}