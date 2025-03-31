from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def optimal_charge_fit_0af9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV Fleet Charging Optimization with dynamic charging priorities.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Total number of EVs in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - order_price (list[float]): Payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): (if any, can be omitted)
        get_state_data_function (callable): (if any, can be omitted)
        base_charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.3.
        base_charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.4.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.3)
    base_charge_ub = kwargs.get('base_charge_ub', 0.4)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    order_price = global_data['order_price']

    # Predict future demand and price trends
    future_demand = np.mean(customer_arrivals[current_step:current_step + 5]) if current_step + 5 < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])
    future_order_price = np.mean(order_price[current_step:current_step + 5]) if current_step + 5 < len(order_price) else np.mean(order_price[current_step:])

    # Adjust thresholds based on future demand and price trends
    charge_lb = max(base_charge_lb, 0.5 - 0.1 * (future_demand / np.max(customer_arrivals)))
    charge_ub = min(base_charge_ub, 0.6 + 0.1 * (future_order_price / np.max(order_price)))

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low time_to_next_availability and battery_soc
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Ensure at least one EV is chosen for charging if possible
    if sum(actions) == 0 and total_chargers > 0:
        idle_ev_indices = [index for index, status in enumerate(operational_status) if status == 0]
        if idle_ev_indices:
            lowest_soc_index = min(idle_ev_indices, key=lambda i: battery_soc[i])
            actions[lowest_soc_index] = 1

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}