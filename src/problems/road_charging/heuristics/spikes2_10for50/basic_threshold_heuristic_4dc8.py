from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_4dc8(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Enhanced Dynamic Threshold Heuristic for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Algorithm-specific data if necessary.
        get_state_data_function (callable): Function to receive the new solution as input and return the state dictionary for new solution.
        base_charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.4.
        base_charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.5.
        min_SoC_threshold (float, optional): Minimum battery SoC threshold for prioritizing charging. Default is 0.1.
        fleet_to_charger_ratio_threshold (int, optional): Threshold for fleet-to-charger ratio to apply aggressive charging logic. Default is 15.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.4)
    base_charge_ub = kwargs.get('base_charge_ub', 0.5)
    min_SoC_threshold = kwargs.get('min_SoC_threshold', 0.1)
    fleet_to_charger_ratio_threshold = kwargs.get('fleet_to_charger_ratio_threshold', 15)

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

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery_soc below min_SoC_threshold
        elif battery_soc[i] <= min_SoC_threshold:
            actions[i] = 1
        # Apply dynamic thresholds based on fleet competition
        elif fleet_size / total_chargers > fleet_to_charger_ratio_threshold:
            # Use weighted scoring to choose EV for charging
            scores = [(i, battery_soc[i] + 0.1 * future_demand - operational_status[i]) for i in range(fleet_size)]
            chosen_ev = min(scores, key=lambda x: x[1])[0]
            actions[chosen_ev] = 1
        elif battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif battery_soc[i] >= base_charge_ub:
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